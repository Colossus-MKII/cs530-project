import airsim
import time
import math
import cv2
import numpy as np

# ==========================================
# Configuration for Survey Mission
# ==========================================
SURVEY_ALTITUDE = -200.0  # Ascend to 200 meters (Negative Z is upwards)
SAFETY_HEIGHT = 5.0       # Any object taller than 5 meters is considered an obstacle

def main():
    print("[Scanner] Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    print("[Scanner] Requesting API control...")
    client.enableApiControl(True)
    client.armDisarm(True)

    # 1. Fly to absolute center (0,0) and ascend to survey altitude
    print(f"[Scanner] Taking off and climbing to {abs(SURVEY_ALTITUDE)} meters...")
    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, SURVEY_ALTITUDE, 10).join()
    time.sleep(3) # Stabilize the drone

    # 2. Point Camera '0' straight down (Pitch = -90 degrees)
    print("[Scanner] Pointing camera downwards...")
    camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-math.pi/2, 0, 0))
    client.simSetCameraPose("0", camera_pose)
    time.sleep(2)

    # 3. Capture both RGB (Scene) and Depth data simultaneously
    print("[Scanner] Capturing RGB and Depth images...")
    responses = client.simGetImages([
        # Request 0: Standard Color Image (for humans)
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        # Request 1: Planar Depth Matrix (for algorithms)
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
    ])

    # ==========================================
    # 4. Process RGB Image (Human-readable Satellite Map)
    # ==========================================
    print("[Scanner] Processing RGB map...")
    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
    # Resize to our standard 1024x1024 radar resolution
    map_visual = cv2.resize(img_rgb, (1024, 1024))
    cv2.imwrite("satellite_map.png", map_visual)

    # ==========================================
    # 5. Process Depth Map (A* Obstacle Matrix)
    # ==========================================
    print("[Scanner] Processing Depth data into Obstacle Matrix...")
    # Convert depth data to a 2D float array (Each pixel represents distance in meters)
    depth_array = airsim.list_to_2d_float_array(responses[1].image_data_float, responses[1].width, responses[1].height)
    depth_resized = cv2.resize(depth_array, (800, 800), interpolation=cv2.INTER_NEAREST)

    # Logic: Drone is at 200m height. 
    # If distance recorded is less than (200 - 5) = 195m, the object is taller than 5m.
    # Therefore, it is an obstacle.
    threshold_distance = abs(SURVEY_ALTITUDE) - SAFETY_HEIGHT
    
    # Create a visual binary matrix: 
    # 0 (Black) for Obstacles, 255 (White) for Free Space (Streets)
    obstacle_matrix = np.where(depth_resized < threshold_distance, 0, 255).astype(np.uint8)
    cv2.imwrite("obstacle_grid.png", obstacle_matrix)
    
    # Save the raw numpy array for mathematical calculations.
    # We keep the standard logic: 1 = Obstacle, 0 = Free Space for A* pathfinding.
    grid_for_astar = np.where(depth_resized < threshold_distance, 1, 0).astype(np.uint8)
    np.save("astar_grid.npy", grid_for_astar)

    print("[Scanner] SUCCESS! Generated:")
    print("  1. 'satellite_map.png' (For radar background)")
    print("  2. 'obstacle_grid.png' (White streets, Black buildings)")
    print("  3. 'astar_grid.npy'    (Raw mathematical matrix for A* algorithm)")

    # 6. Return to Base
    print("[Scanner] Returning to base...")
    client.simSetCameraPose("0", airsim.Pose()) # Reset camera
    client.moveToPositionAsync(0, 0, -5, 10).join()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("[Scanner] Mission Complete.")

if __name__ == "__main__":
    main()