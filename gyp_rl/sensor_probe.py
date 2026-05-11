import airsim
import time
import numpy as np


# =========================
# Basic utility functions
# =========================

def connect_drone(enable_api_control=True, arm=False):
    """
    Connect to AirSim drone and return client.
    """
    client = airsim.MultirotorClient()
    client.confirmConnection()

    if enable_api_control:
        client.enableApiControl(True)

    if arm:
        client.armDisarm(True)

    return client


def vec3_to_tuple(v):
    return (v.x_val, v.y_val, v.z_val)


def quat_to_tuple(q):
    return (q.w_val, q.x_val, q.y_val, q.z_val)


def safe_call(func, default=None):
    try:
        return func()
    except Exception as e:
        return default


# =========================
# Core drone state
# =========================

def get_drone_state(client):
    """
    Get basic multirotor state.
    """
    state = client.getMultirotorState()
    kin = state.kinematics_estimated

    return {
        "position": vec3_to_tuple(kin.position),
        "linear_velocity": vec3_to_tuple(kin.linear_velocity),
        "linear_acceleration": vec3_to_tuple(kin.linear_acceleration),
        "orientation_quaternion": quat_to_tuple(kin.orientation),
        "angular_velocity": vec3_to_tuple(kin.angular_velocity),
        "landed_state": state.landed_state,
        "timestamp": state.timestamp
    }


def get_collision_info(client):
    """
    Get collision information.
    """
    collision = client.simGetCollisionInfo()

    return {
        "has_collided": collision.has_collided,
        "object_name": collision.object_name,
        "impact_point": vec3_to_tuple(collision.impact_point),
        "normal": vec3_to_tuple(collision.normal),
        "penetration_depth": collision.penetration_depth
    }


# =========================
# Default sensors
# =========================

def get_imu_data(client):
    imu = client.getImuData()
    return {
        "angular_velocity": vec3_to_tuple(imu.angular_velocity),
        "linear_acceleration": vec3_to_tuple(imu.linear_acceleration),
        "orientation": quat_to_tuple(imu.orientation),
        "timestamp": imu.time_stamp
    }


def get_gps_data(client):
    gps = client.getGpsData()
    return {
        "is_valid": gps.is_valid,
        "latitude": gps.gnss.geo_point.latitude,
        "longitude": gps.gnss.geo_point.longitude,
        "altitude": gps.gnss.geo_point.altitude,
        "velocity": vec3_to_tuple(gps.gnss.velocity),
        "fix_type": gps.gnss.fix_type,
        "eph": gps.gnss.eph,
        "epv": gps.gnss.epv,
        "timestamp": gps.time_stamp
    }


def get_barometer_data(client):
    baro = client.getBarometerData()
    return {
        "altitude": baro.altitude,
        "pressure": baro.pressure,
        "qnh": baro.qnh,
        "timestamp": baro.time_stamp
    }


def get_magnetometer_data(client):
    mag = client.getMagnetometerData()
    return {
        "magnetic_field_body": vec3_to_tuple(mag.magnetic_field_body),
        "timestamp": mag.time_stamp
    }

# =========================
# DistanceFront
# =========================
def get_distance_sensor_data(client, sensor_name="DistanceFront"):
    try:
        data = client.getDistanceSensorData(sensor_name)
        return {
            "available": True,
            "distance": data.distance,
            "min_distance": data.min_distance,
            "max_distance": data.max_distance,
            "relative_pose_position": vec3_to_tuple(data.relative_pose.position),
            "relative_pose_orientation": quat_to_tuple(data.relative_pose.orientation),
            "timestamp": data.time_stamp
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

# =========================
# Lidar
# =========================

def get_lidar_points(client, lidar_name="LidarSensor1"):
    """
    Return lidar point cloud as Nx3 numpy array.
    If lidar does not exist or has no points, return empty array.
    """
    try:
        lidar = client.getLidarData(lidar_name)

        if len(lidar.point_cloud) < 3:
            return np.empty((0, 3), dtype=np.float32)

        points = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
        return points

    except Exception:
        return np.empty((0, 3), dtype=np.float32)


def get_lidar_summary(client, lidar_name="LidarSensor1"):
    points = get_lidar_points(client, lidar_name)

    if points.shape[0] == 0:
        return {
            "available": False,
            "num_points": 0,
            "min_distance": None,
            "max_distance": None
        }

    distances = np.linalg.norm(points, axis=1)

    return {
        "available": True,
        "num_points": int(points.shape[0]),
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "first_points": points[:5].tolist()
    }


def get_lidar_sector_distances(client, lidar_name="LidarSensor1"):
    """
    Convert lidar point cloud into simple distances for RL state.
    Assumption:
    x = forward
    y = left/right
    z = up/down
    """
    points = get_lidar_points(client, lidar_name)

    if points.shape[0] == 0:
        return {
            "front": 100.0,
            "left": 100.0,
            "right": 100.0
        }

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    horizontal_dist = np.sqrt(x ** 2 + y ** 2)
    angles = np.degrees(np.arctan2(y, x))

    # Ignore ground / extreme vertical points
    height_mask = np.abs(z) < 2.0

    front_mask = (x > 0) & height_mask & (angles >= -20) & (angles <= 20)
    left_mask = (x > 0) & height_mask & (angles > 20) & (angles <= 90)
    right_mask = (x > 0) & height_mask & (angles >= -90) & (angles < -20)

    def min_dist(mask):
        if np.any(mask):
            return float(np.min(horizontal_dist[mask]))
        return 100.0

    return {
        "front": min_dist(front_mask),
        "left": min_dist(left_mask),
        "right": min_dist(right_mask)
    }

def get_lidar_36x3_sector_distances(
    client,
    lidar_name="LidarSensor1",
    max_range=15.0,
    h_sectors=36,
    v_layers=3,
    vertical_fov=(-30.0, 30.0)
):
    points = get_lidar_points(client, lidar_name)

    if points.shape[0] == 0:
        return np.full((v_layers, h_sectors), max_range, dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    distances_3d = np.linalg.norm(points, axis=1)

    horizontal_angles = np.degrees(np.arctan2(y, x))
    horizontal_angles = (horizontal_angles + 360) % 360

    xy_dist = np.sqrt(x ** 2 + y ** 2)
    vertical_angles = np.degrees(np.arctan2(z, xy_dist))

    valid_mask = (
        (distances_3d > 0.1) &
        (distances_3d <= max_range) &
        (vertical_angles >= vertical_fov[0]) &
        (vertical_angles <= vertical_fov[1])
    )

    grid = np.full((v_layers, h_sectors), max_range, dtype=np.float32)

    valid_h_angles = horizontal_angles[valid_mask]
    valid_v_angles = vertical_angles[valid_mask]
    valid_distances = distances_3d[valid_mask]

    h_bin_size = 360.0 / h_sectors
    v_min, v_max = vertical_fov
    v_bin_size = (v_max - v_min) / v_layers

    for h_angle, v_angle, dist in zip(valid_h_angles, valid_v_angles, valid_distances):
        h_idx = int(h_angle // h_bin_size)
        v_idx = int((v_angle - v_min) // v_bin_size)

        h_idx = min(max(h_idx, 0), h_sectors - 1)
        v_idx = min(max(v_idx, 0), v_layers - 1)

        grid[v_idx, h_idx] = min(grid[v_idx, h_idx], dist)

    return grid


# =========================
# Depth image
# =========================

def get_depth_image(client, camera_name="0"):
    """
    Return DepthPlanar image as HxW numpy array.
    """
    try:
        responses = client.simGetImages([
            airsim.ImageRequest(
                camera_name=camera_name,
                image_type=airsim.ImageType.DepthPlanar,
                pixels_as_float=True,
                compress=False
            )
        ])

        if len(responses) == 0 or responses[0].width == 0:
            return None

        response = responses[0]
        depth = np.array(response.image_data_float, dtype=np.float32)
        depth = depth.reshape(response.height, response.width)

        return depth

    except Exception:
        return None


def get_depth_summary(client, camera_name="0"):
    depth = get_depth_image(client, camera_name)

    if depth is None:
        return {
            "available": False,
            "shape": None,
            "min": None,
            "max": None,
            "mean": None,
            "center": None
        }

    h, w = depth.shape

    return {
        "available": True,
        "shape": depth.shape,
        "min": float(np.nanmin(depth)),
        "max": float(np.nanmax(depth)),
        "mean": float(np.nanmean(depth)),
        "center": float(depth[h // 2, w // 2])
    }


def get_depth_sector_distances(client, camera_name="0"):
    """
    Extract front / left / right distance from depth image.
    """
    depth = get_depth_image(client, camera_name)

    if depth is None:
        return {
            "front": 100.0,
            "left": 100.0,
            "right": 100.0
        }

    depth = np.where(depth > 1000, np.nan, depth)

    h, w = depth.shape
    y1, y2 = int(h * 0.4), int(h * 0.6)

    left_region = depth[y1:y2, int(w * 0.1):int(w * 0.3)]
    front_region = depth[y1:y2, int(w * 0.4):int(w * 0.6)]
    right_region = depth[y1:y2, int(w * 0.7):int(w * 0.9)]

    def safe_min(region):
        if np.all(np.isnan(region)):
            return 100.0
        return float(np.nanmin(region))

    return {
        "left": safe_min(left_region),
        "front": safe_min(front_region),
        "right": safe_min(right_region)
    }


# =========================
# RGB scene image
# =========================

def get_scene_image(client, camera_name="0"):
    """
    Return RGB scene image as HxWx3 numpy array.
    """
    try:
        responses = client.simGetImages([
            airsim.ImageRequest(
                camera_name=camera_name,
                image_type=airsim.ImageType.Scene,
                pixels_as_float=False,
                compress=False
            )
        ])

        if len(responses) == 0 or responses[0].width == 0:
            return None

        response = responses[0]
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = img.reshape(response.height, response.width, 3)

        return img

    except Exception:
        return None


# =========================
# All-in-one data reader
# =========================

def get_all_sensor_data(client):
    """
    Read all available drone data.
    This is the main function other files can call.
    """
    return {
        "drone_state": safe_call(lambda: get_drone_state(client)),
        "collision": safe_call(lambda: get_collision_info(client)),
        "imu": safe_call(lambda: get_imu_data(client)),
        "gps": safe_call(lambda: get_gps_data(client)),
        "barometer": safe_call(lambda: get_barometer_data(client)),
        "magnetometer": safe_call(lambda: get_magnetometer_data(client)),
        "distance_front": get_distance_sensor_data(client),
        "lidar_summary": get_lidar_summary(client),
        "lidar_sector_distances": get_lidar_sector_distances(client),
        "depth_summary": get_depth_summary(client),
        "depth_sector_distances": get_depth_sector_distances(client)
    }


# =========================
# Pretty print
# =========================

def print_dict(title, data):
    print(f"\n========== {title} ==========")
    if data is None:
        print("None")
        return

    for key, value in data.items():
        print(f"{key}: {value}")


def print_all_sensor_data(client):
    data = get_all_sensor_data(client)

    for key, value in data.items():
        print_dict(key.upper(), value)


# =========================
# Run directly
# =========================

def main():
    print("Connecting to AirSim...")
    client = connect_drone(enable_api_control=True, arm=False)
    print("Connected.")

    print_all_sensor_data(client)

    print("\nSensor data read completed.")


if __name__ == "__main__":
    #main()
    client = connect_drone(enable_api_control=True, arm=True)

    print("Taking off...")
    client.takeoffAsync().join()

    print("Moving to 3m altitude...")
    client.moveToPositionAsync(0, 0, -3, 2).join()

    time.sleep(1.0)



    # optional: print current position
    state = get_drone_state(client)
    print("Current position:", state["position"])