import airsim
import cv2
import numpy as np
import math
import os

# ==========================================
# 1. Map Coordinates and Geofence Boundaries
# Defines the absolute 250m x 250m valid operational area.
# Exceeding this boundary in RL training is equivalent to a fatal crash.
# Assuming Base(0,0) is roughly the center of the neighborhood.
# ==========================================
GEOFENCE = {
    "x_min": -175, "x_max": 175,
    "y_min": -175, "y_max": 175
}

# ==========================================
# 2. Coordinate Mapper (AirSim NED -> OpenCV Screen)
# ==========================================
SCALE = 2.56        # 1 meter in AirSim = 2 pixels on screen
OFFSET_X = 512   # Center X of the 600x600 canvas
OFFSET_Y = 512   # Center Y of the 600x600 canvas

def world_to_pixel(airsim_x, airsim_y):
    # AirSim Y (East) maps to OpenCV X (Right)
    # AirSim X (North) maps to OpenCV Y (Down, requiring negation)
    px = int(airsim_y * SCALE + OFFSET_X)
    py = int(-airsim_x * SCALE + OFFSET_Y)
    return px, py

def main():
    print("[Telemetry] Initializing Global Map with Geofence...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    cv2.namedWindow("Global Delivery Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Global Delivery Map", 600, 600)
    print("[Telemetry] Map is live. Press 'Q' to exit.")
    
    path_history = []
    
    # Load Satellite Background if available
    bg_image_path = "satellite_map.png"
    has_bg = os.path.exists(bg_image_path)
    
    if has_bg:
        bg_image = cv2.imread(bg_image_path)
        bg_image = cv2.resize(bg_image, (1024, 1024)) 
        print("[Telemetry] Satellite background loaded.")
    else:
        print("[Telemetry] WARNING: 'satellite_map.png' not found. Using black canvas.")

    while True:
        # 1. Initialize Canvas
        if has_bg:
            canvas = bg_image.copy()
        else:
            canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
            
        # 2. Render Geofence (The 250x250m absolute boundary)
        gf_p1 = world_to_pixel(GEOFENCE["x_min"], GEOFENCE["y_min"])
        gf_p2 = world_to_pixel(GEOFENCE["x_max"], GEOFENCE["y_max"])
        # Draw a thick red boundary box to represent the edge of the world
        cv2.rectangle(canvas, gf_p1, gf_p2, (0, 0, 255), 3)
        cv2.putText(canvas, "GEOFENCE BOUNDARY", (gf_p1[0] + 5, gf_p1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        # 3. Render Base Station (0, 0)
        origin_px, origin_py = world_to_pixel(0, 0)
        cv2.drawMarker(canvas, (origin_px, origin_py), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(canvas, "Base(0,0)", (origin_px+10, origin_py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 4. Fetch Real-time Drone Telemetry
        state = client.getMultirotorState()
        x = state.kinematics_estimated.position.x_val
        y = state.kinematics_estimated.position.y_val
        z = state.kinematics_estimated.position.z_val
        vx = state.kinematics_estimated.linear_velocity.x_val
        vy = state.kinematics_estimated.linear_velocity.y_val
        speed = math.sqrt(vx**2 + vy**2)
        
        # 5. Render Trajectory History
        drone_px, drone_py = world_to_pixel(x, y)
        
        # Record trajectory only when airborne (Z is negative upwards)
        if z < -1: 
            path_history.append((drone_px, drone_py))
            
        # Draw the continuous trajectory line (Orange)
        if len(path_history) > 1:
            pts = np.array(path_history, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 165, 255), thickness=2)

        # 6. Render Current Drone Position and Heading
        vec_scale = 5 # Amplify speed vector
        end_px, end_py = world_to_pixel(x + vx * vec_scale, y + vy * vec_scale)
        
        # Change drone color to RED if it violates the Geofence
        is_violating_geofence = (x < GEOFENCE["x_min"] or x > GEOFENCE["x_max"] or 
                                 y < GEOFENCE["y_min"] or y > GEOFENCE["y_max"])
        drone_color = (0, 0, 255) if is_violating_geofence else (0, 255, 0)
        
        if speed > 0.5:
            cv2.arrowedLine(canvas, (drone_px, drone_py), (end_px, end_py), drone_color, 2, tipLength=0.3)
        cv2.circle(canvas, (drone_px, drone_py), 6, drone_color, -1)

        # 7. Render Telemetry HUD
        info_text = [
            f"Pos: ({x:.1f}, {y:.1f})",
            f"Alt: {-z:.1f} m",
            f"Speed: {speed:.1f} m/s",
            f"Status: {'OUT OF BOUNDS!' if is_violating_geofence else 'SAFE'}"
        ]
        
        # HUD background for readability
        cv2.rectangle(canvas, (10, 10), (280, 130), (0, 0, 0), -1)
        for i, text in enumerate(info_text):
            text_color = (0, 0, 255) if (i == 3 and is_violating_geofence) else (0, 255, 255)
            cv2.putText(canvas, text, (20, 35 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # 8. Display Frame
        cv2.imshow("Global Delivery Map", canvas)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            print("[Telemetry] Shutting down global map.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()