import airsim
import time

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("[TestPilot] Taking off...")
    client.takeoffAsync().join()

    # Move to starting point (Cruising altitude: 10m)
    print("[TestPilot] Climbing to 10m altitude...")
    client.moveToPositionAsync(0, 0, -10, 5).join()

    # Flight path: A 40-meter square
    waypoints = [
        (40, 0),   # North 40m
        (40, 40),  # Then East 40m
        (0, 40),   # Then South 40m
        (0, 0)     # Back to Base
    ]

    for x, y in waypoints:
        print(f"[TestPilot] Flying to: X={x}, Y={y}")
        client.moveToPositionAsync(x, y, -10, 5).join()
        time.sleep(1)

    print("[TestPilot] Mission complete. Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

if __name__ == "__main__":
    main()