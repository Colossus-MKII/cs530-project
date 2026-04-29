import airsim
import cv2
import numpy as np
import heapq
import math
import time

# ==========================================
# 1. Global Configurations & Map Data
# ==========================================
print("[System] Loading A* Obstacle Grid (1024x1024)...")
GRID = np.load("astar_grid.npy")        
BG_IMAGE = cv2.imread("satellite_map.png") 
SCALE = 2.56
OFFSET = 512
CRUISE_ALTITUDE = -15.0  # meters
FLIGHT_SPEED = 6.0       # m/s

def world_to_pixel(x, y):
    col = int(y * SCALE + OFFSET)
    row = int(-x * SCALE + OFFSET)
    return row, col

def pixel_to_world(row, col):
    x = -(row - OFFSET) / SCALE
    y = (col - OFFSET) / SCALE
    return x, y

# ==========================================
# 2. A* Pathfinding Logic (The Brain)
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
            neighbor = current[0] + i, current[1] + j            
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
# 3. UI Interaction & State Management
# ==========================================
points = []          
found_path = []      
trigger_astar = False # State flag to solve the UX lag issue

def mouse_click(event, x, y, flags, param):
    global points, found_path, trigger_astar
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            if GRID[y][x] == 1:
                print("[Warning] Obstacle! Click on the road.")
                return
            points.append((y, x))
            if len(points) == 1:
                print(f"[UI] START point set at Pixel({x}, {y}).")
            elif len(points) == 2:
                print(f"[UI] END point set at Pixel({x}, {y}).")
                # We set the trigger but DON'T run A* here to allow UI to render the red dot
                trigger_astar = True 

# ==========================================
# 4. Flight Execution (The Body)
# ==========================================
def start_flight(pixel_path):
    if not pixel_path: return
    
    print("\n[System] Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    # [FIX] Reset the drone to clear previous physics/collision states
    # This ensures simSetVehiclePose works every time, especially after 'R'
    print("[Drone] Hard Resetting vehicle state...")
    client.reset() 
    time.sleep(0.5)
    
    # 1. Teleport to START
    start_row, start_col = pixel_path[0]
    start_world_x, start_world_y = pixel_to_world(start_row, start_col)
    
    print(f"[Drone] Teleporting to Map START: ({start_world_x:.1f}, {start_world_y:.1f})")
    start_pose = airsim.Pose(airsim.Vector3r(start_world_x, start_world_y, -0.5), # Slightly above ground
                             airsim.to_quaternion(0, 0, 0))
    client.simSetVehiclePose(start_pose, ignore_collision=True)
    time.sleep(1) 
    
    # 2. Takeoff
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[Drone] Taking off...")
    client.takeoffAsync().join()

    # 3. Waypoint Downsampling
    waypoints = []
    for i in range(0, len(pixel_path), 15):
        r, c = pixel_path[i]
        wx, wy = pixel_to_world(r, c)
        waypoints.append((wx, wy))
    last_x, last_y = pixel_to_world(pixel_path[-1][0], pixel_path[-1][1])
    waypoints.append((last_x, last_y))

    # 4. Mission Execution
    print(f"[Drone] Climbing to cruise altitude: {abs(CRUISE_ALTITUDE)}m")
    client.moveToPositionAsync(start_world_x, start_world_y, CRUISE_ALTITUDE, 5).join()

    print(f"[Drone] Navigating {len(waypoints)} waypoints...")
    for wx, wy in waypoints:
        client.moveToPositionAsync(wx, wy, CRUISE_ALTITUDE, FLIGHT_SPEED).join()
    
    print("[Drone] Arrived. Descending for landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("[Mission] Delivery completed.\n")

# ==========================================
# 5. Main Loop
# ==========================================
def main():
    global trigger_astar, found_path, points
    window_name = "CS530 Global Delivery Map"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_click)
    
    print("\n" + "="*50)
    print("      CS530 AUTONOMOUS DELIVERY SYSTEM      ")
    print("="*50)
    print(" - LEFT CLICK: Set Start (Green) and End (Red)")
    print(" - PRESS 'S': Launch Mission (Reset + Teleport + Fly)")
    print(" - PRESS 'R': Reset Map & Clear States")
    print(" - PRESS 'Q': Quit")
    print("="*50 + "\n")

    while True:
        # 1. Update Canvas
        canvas = BG_IMAGE.copy()
        if len(points) >= 1:
            cv2.circle(canvas, (points[0][1], points[0][0]), 6, (0, 255, 0), -1) 
            cv2.putText(canvas, "START", (points[0][1]+10, points[0][0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(points) == 2:
            cv2.circle(canvas, (points[1][1], points[1][0]), 6, (0, 0, 255), -1) 
            cv2.putText(canvas, "END", (points[1][1]+10, points[1][0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if found_path:
            path_pts = np.array([(p[1], p[0]) for p in found_path], np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [path_pts], False, (0, 165, 255), 3) 

        # 2. Render Window
        cv2.imshow(window_name, canvas)
        
        # 3. Handle Heavy Logic after Rendering
        if trigger_astar:
            # Force UI refresh to show the red dot before blocking for calculation
            cv2.waitKey(1) 
            print("[Brain] Planning route... please wait.")
            found_path = astar(points[0], points[1])
            trigger_astar = False
            if found_path:
                print(f"[Brain] Path found ({len(found_path)} nodes). Press 'S' to Fly.")
            else:
                print("[Brain] ERROR: Destination unreachable.")

        # 4. Handle Keyboard
        key = cv2.waitKey(33) & 0xFF
        if key == ord('s') and found_path:
            start_flight(found_path)
        elif key == ord('r'):
            print("[System] Resetting map and drone states.")
            points.clear()
            found_path.clear()
            trigger_astar = False
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()