import time
import cv2
import airsim
import numpy as np

from sensor_probe import (
    connect_drone,
    get_drone_state,
    get_lidar_18_sector_distances
)

SCALE = 2.56
OFFSET_X = 512
OFFSET_Y = 512

SATELLITE_MAP_PATH = "../satellite_map.png"
CRUISE_ALTITUDE = -3.0


def pixel_to_world(px, py):
    """
    Convert OpenCV pixel coordinate to AirSim world coordinate.
    """
    airsim_y = (px - OFFSET_X) / SCALE
    airsim_x = -(py - OFFSET_Y) / SCALE
    return airsim_x, airsim_y


def world_to_pixel(airsim_x, airsim_y):
    px = int(airsim_y * SCALE + OFFSET_X)
    py = int(-airsim_x * SCALE + OFFSET_Y)
    return px, py


def set_drone_start_and_takeoff(client, x, y):
    print(f"\nSelected start position: x={x:.2f}, y={y:.2f}")

    client.reset()
    time.sleep(1.0)

    client.enableApiControl(True)
    client.armDisarm(True)

    # Put drone near the ground at selected location
    pose = airsim.Pose(
        airsim.Vector3r(x, y, -0.2),
        airsim.to_quaternion(0, 0, 0)
    )
    client.simSetVehiclePose(pose, ignore_collision=True)
    time.sleep(0.5)

    print("Taking off...")
    client.takeoffAsync().join()

    print("Moving to 3m altitude...")
    client.moveToPositionAsync(x, y, CRUISE_ALTITUDE, 2).join()

    time.sleep(1.0)

    state = get_drone_state(client)
    print("Current position:", state["position"])

    sectors = get_lidar_18_sector_distances(client)
    print("18-sector lidar distances:")
    print(sectors)
    print("shape:", sectors.shape)


def main():
    client = connect_drone(enable_api_control=True, arm=False)

    img = cv2.imread(SATELLITE_MAP_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot find {SATELLITE_MAP_PATH}")

    img = cv2.resize(img, (1024, 1024))
    display = img.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal display

        if event == cv2.EVENT_LBUTTONDOWN:
            airsim_x, airsim_y = pixel_to_world(x, y)

            display = img.copy()
            cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"Start ({airsim_x:.1f}, {airsim_y:.1f})",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            cv2.imshow("Click Start Position", display)

            set_drone_start_and_takeoff(client, airsim_x, airsim_y)

    cv2.namedWindow("Click Start Position", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Click Start Position", 1024, 1024)
    cv2.setMouseCallback("Click Start Position", on_mouse)

    print("Click on the satellite map to set drone start position.")
    print("Press q to quit.")

    while True:
        cv2.imshow("Click Start Position", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()