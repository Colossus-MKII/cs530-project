import airsim
import time

def main():
    # 1. Establish connection to the AirSim physics engine
    print("Attempting to connect to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected successfully!")

    # 2. Take control and unlock the propellers (Arm)
    print("Taking API control and arming the drone...")
    client.enableApiControl(True)
    client.armDisarm(True)

    # 3. Takeoff vertically
    print("Taking off!")
    # .join() blocks the execution until the takeoff is complete
    client.takeoffAsync().join() 

    # 4. Execute flight mission (A mini test for future A* integration)
    # Note: In the NED coordinate system, the Z-axis points downwards. 
    # Therefore, -10 means flying 10 meters upwards.
    print("Climbing to a safe cruising altitude of 10 meters...")
    client.moveToPositionAsync(0, 0, -10, 5).join() # 5 is the velocity in m/s

    print("Flying forward 20 meters...")
    client.moveToPositionAsync(20, 0, -10, 5).join()

    print("Target reached. Hovering for 3 seconds...")
    time.sleep(3)

    # 5. Land the drone
    print("Mission accomplished. Preparing to land...")
    client.landAsync().join()

    # 6. Lock propellers and release API control
    print("Disarming the drone and releasing control.")
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("CS 530 first test flight completed successfully!")

if __name__ == "__main__":
    main()