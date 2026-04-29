import airsim
import cv2
import numpy as np
import heapq
import math
import time
import os

# ==========================================
# [Architecture] Global A* + Local APF/RL
# ==========================================
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("[System] stable-baselines3 not found. Defaulting to APF Baseline.")

# Map Configuration
GRID = np.load("astar_grid.npy")        
BG_IMAGE = cv2.imread("satellite_map.png") 
SCALE = 2.56
OFFSET = 512

# Flight & Control Parameters
CRUISE_ALTITUDE = -10.0  # Meters (Negative Z is UP)
MAX_SPEED = 6.0          
CONTROL_HZ = 10.0        
WP_TOLERANCE = 2.5       
APF_TRIGGER_DIST = 0.25  # Trigger APF if obstacle < 10m

# FSM States
STATE_FOLLOW = "FOLLOWING_A_STAR"
STATE_AVOID = "APF_AVOIDANCE"
STATE_RECOVER = "PATH_RECOVERY"

# RL Model Setup
MODEL_PATH = "drone_rl_model.zip"
rl_model = None
if RL_AVAILABLE and os.path.exists(MODEL_PATH):
    rl_model = PPO.load(MODEL_PATH)
    print(f"[System] RL Model loaded: {MODEL_PATH}")
else:
    print("[System] Operating with APF Heuristic Baseline.")

def world_to_pixel(x, y):
    return int(-x * SCALE + OFFSET), int(y * SCALE + OFFSET)

def pixel_to_world(row, col):
    return -(row - OFFSET) / SCALE, (col - OFFSET) / SCALE

# ==========================================
# 1. Macro-Brain: A* Planner
# ==========================================
def get_heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def astar(start, goal):
    neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: get_heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)            
            if 0 <= neighbor[0] < 1024 and 0 <= neighbor[1] < 1024:
                if GRID[neighbor[0]][neighbor[1]] == 1: continue
            else: continue
                
            move_cost = math.sqrt(i**2 + j**2)
            tentative_g_score = gscore[current] + move_cost
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0): continue
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [n[1] for n in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = gscore[neighbor] + get_heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return None

# ==========================================
# 2. Perception & Safety Check
# ==========================================
def parse_lidar(lidar_data):
    sectors = np.ones(8) * 40.0 
    if len(lidar_data.point_cloud) < 3: return sectors / 40.0 
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    for p in points:
        x, y, z = p[0], p[1], p[2]
        if abs(z) > 2.0: continue 
        dist = math.sqrt(x**2 + y**2)
        angle = math.atan2(y, x) 
        idx = int(((angle + math.pi) / (2 * math.pi)) * 8) % 8
        if dist < sectors[idx]: sectors[idx] = dist
    return sectors / 40.0 

def safe_landing_check(client):
    """Detects absolute ground elevation to block landing on roofs."""
    print("\n[Safety] Validating Ground Integrity...")
    dist_data = client.getDistanceSensorData(distance_sensor_name="DistanceSensor_Down")
    current_z = client.getMultirotorState().kinematics_estimated.position.z_val
    # Absolute Altitude = Z_pos + Raycast_Distance
    surface_alt = current_z + dist_data.distance 
    print(f"[Safety] Detected Absolute Surface Z: {surface_alt:.2f}m")
    
    if abs(surface_alt) > 0.8:
        print("[Critical] LANDING REJECTED: Surface is elevated (Roof/Obstacle).")
        return False
    print("[Safety] LANDING PERMITTED: Flat ground confirmed.")
    return True

# ==========================================
# 3. Flight Controller (FSM)
# ==========================================
def start_flight(pixel_path):
    if not pixel_path: return
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.reset() 
    time.sleep(0.5)
    
    # Path processing
    waypoints = []
    for i in range(0, len(pixel_path), 15):
        r, c = pixel_path[i]
        wx, wy = pixel_to_world(r, c)
        waypoints.append((wx, wy))
    last_x, last_y = pixel_to_world(pixel_path[-1][0], pixel_path[-1][1])
    waypoints.append((last_x, last_y))

    # Deployment
    start_x, start_y = waypoints[0]
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(start_x, start_y, -0.5), airsim.to_quaternion(0,0,0)), True)
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # --- PHASE 1: Safe Takeoff & Calibration ---
    print("[Drone] Takeoff initiated. Establishing physics baseline...")
    client.takeoffAsync().join()
    # Force climb to 5m to ignore ground jitter
    client.moveToPositionAsync(start_x, start_y, -5.0, 3).join() 
    
    # [FIX] Get the BASELINE time_stamp instead of non-existent collision_count
    # This ignores any collisions that occurred before this stable moment.
    init_coll = client.simGetCollisionInfo()
    baseline_time = init_coll.time_stamp
    print(f"[Drone] Safety height reached. Baseline Timestamp: {baseline_time}")
    
    client.moveToPositionAsync(start_x, start_y, CRUISE_ALTITUDE, 5).join()

    # --- PHASE 2: Navigation Loop ---
    wp_idx = 1
    fsm_state = STATE_FOLLOW
    deviation_progress = float('inf')
    
    print("\n[Mission] Hybrid Navigation Active.")

    while wp_idx < len(waypoints):
        state = client.getMultirotorState()
        pos, vel = state.kinematics_estimated.position, state.kinematics_estimated.linear_velocity
        target_x, target_y = waypoints[wp_idx]
        dist_to_wp = math.hypot(target_x - pos.x_val, target_y - pos.y_val)
        
        # Sense
        lidar_data = client.getLidarData(lidar_name="LidarSensor1")
        obs_sectors = parse_lidar(lidar_data)
        min_lidar = np.min(obs_sectors)

        # REFINED COLLISION DETECTION (Timestamp-Based)
        curr_coll = client.simGetCollisionInfo()
        # Trigger ONLY if has_collided is true AND it's a NEW event
        if curr_coll.has_collided and curr_coll.time_stamp > baseline_time:
            obj_name = curr_coll.object_name.lower()
            # Glitch Filtering: Ignore persistent road contacts if high up
            if ("road" in obj_name or "landscape" in obj_name) and pos.z_val < -2.0:
                baseline_time = curr_coll.time_stamp # Update baseline to clear flag
            else:
                print(f"\n[Critical] MISSION FAILURE: New collision with {curr_coll.object_name}!")
                client.armDisarm(False)
                return

        # --- FSM Transitions ---
        if min_lidar < APF_TRIGGER_DIST and fsm_state == STATE_FOLLOW:
            fsm_state = STATE_AVOID
            deviation_progress = math.hypot(last_x - pos.x_val, last_y - pos.y_val)
            print("[FSM] LOCAL THREAT: Transitioning to APF Avoidance.")

        elif fsm_state == STATE_AVOID and min_lidar >= APF_TRIGGER_DIST + 0.05:
            fsm_state = STATE_RECOVER
            print("[FSM] Clearing obstacle. Optimizing reentry...")
            best_i = wp_idx
            for i in range(wp_idx, len(waypoints)):
                h = math.hypot(last_x - waypoints[i][0], last_y - waypoints[i][1])
                if h < deviation_progress - 2.0:
                    best_i = i
                    break
            wp_idx = best_i
            fsm_state = STATE_FOLLOW
            print(f"[FSM] Resuming Global A* from Waypoint {wp_idx}.")
            continue

        elif fsm_state == STATE_FOLLOW and dist_to_wp < WP_TOLERANCE:
            wp_idx += 1
            continue

        # --- Velocity Logic ---
        vx, vy = 0.0, 0.0
        if fsm_state == STATE_FOLLOW:
            s = min(MAX_SPEED, dist_to_wp)
            vx, vy = (target_x-pos.x_val)/dist_to_wp*s, (target_y-pos.y_val)/dist_to_wp*s
        elif fsm_state == STATE_AVOID:
            # APF Baseline
            dist_g = math.hypot(last_x - pos.x_val, last_y - pos.y_val)
            att_x, att_y = (last_x-pos.x_val)/dist_g*MAX_SPEED, (last_y-pos.y_val)/dist_g*MAX_SPEED
            rep_x, rep_y = 0.0, 0.0
            for i, d in enumerate(obs_sectors):
                if d < APF_TRIGGER_DIST:
                    ang = (i/8.0)*2*math.pi-math.pi
                    mag = (APF_TRIGGER_DIST - d) * 45.0
                    rep_x -= math.cos(ang)*mag
                    rep_y -= math.sin(ang)*mag
            vx, vy = att_x + rep_x, att_y + rep_y

        # Execute Command
        client.moveByVelocityAsync(vx, vy, 0, 0.1, 
                                   drivetrain=airsim.DrivetrainType.ForwardOnly, 
                                   yaw_mode=airsim.YawMode(False, 0)).join()

    # --- PHASE 3: Landing ---
    print("\n[Mission] Arrival. Executing Landing Protocol...")
    client.moveToPositionAsync(last_x, last_y, -2.0, 3).join() 
    time.sleep(2) 
    
    if safe_landing_check(client):
        client.landAsync().join()
        print("[Mission] Delivery Successful.")
    else:
        print("[Mission] Delivery Safety Failure: Obstruction at landing site.")
        client.moveToPositionAsync(last_x, last_y, CRUISE_ALTITUDE, 5).join()

    client.armDisarm(False); client.enableApiControl(False)

# ==========================================
# 4. GUI & Entry Point
# ==========================================
pts, path, trigger = [], [], False

def mouse_event(ev, x, y, fl, pr):
    global pts, path, trigger
    if ev == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < 2:
            if GRID[y][x] == 1: return
            pts.append((y, x))
            if len(pts) == 2: trigger = True 

def main():
    global trigger, path, pts
    win_name = "Hybrid Navigator"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 600, 600)
    cv2.moveWindow(win_name, 50, 50)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_event)
    print("\n" + "="*50 + "\n   SYSTEM READY: CLICK START AND END POINTS\n" + "="*50)

    while True:
        frame = BG_IMAGE.copy()
        if len(pts) >= 1: cv2.circle(frame, (pts[0][1], pts[0][0]), 6, (0, 255, 0), -1)
        if len(pts) == 2: cv2.circle(frame, (pts[1][1], pts[1][0]), 6, (0, 0, 255), -1)
        if path:
            ln = np.array([(p[1], p[0]) for p in path], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [ln], False, (0, 165, 255), 3) 
        cv2.imshow(win_name, frame)
        
        if trigger:
            cv2.waitKey(1)
            path = astar(pts[0], pts[1])
            trigger = False
            if path: print("[System] Global path ready. Press 'S' to Start.")

        key = cv2.waitKey(33) & 0xFF
        if key == ord('s') and path: start_flight(path)
        elif key == ord('r'): pts, path, trigger = [], [], False
        elif key == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()