import airsim
import numpy as np
import time
import matplotlib.pyplot as plt

from gyp_rl.sensor_probe import get_lidar_36x3_sector_distances


def get_lidar_test_data(
    altitude=3.0,
    lidar_name="LidarSensor1",
    vehicle_name=""
):
    """
    altitude: 正数，表示飞到几米高度，例如 3.0 = 3m
    AirSim 里面 z 是负数，所以 3m 高度对应 z = -3.0
    """

    client = airsim.MultirotorClient()
    client.confirmConnection()

    client.enableApiControl(True, vehicle_name=vehicle_name)
    client.armDisarm(True, vehicle_name=vehicle_name)

    print("Taking off...")
    client.takeoffAsync(vehicle_name=vehicle_name).join()

    target_z = -abs(altitude)

    print(f"Moving to altitude: {altitude} m  (AirSim z = {target_z})")
    client.moveToPositionAsync(
        0,
        0,
        target_z,
        2,
        vehicle_name=vehicle_name
    ).join()

    time.sleep(2.0)

    state = client.getMultirotorState(vehicle_name=vehicle_name)
    pos = state.kinematics_estimated.position

    print("\n========== Drone Position ==========")
    print(f"x: {pos.x_val:.3f}")
    print(f"y: {pos.y_val:.3f}")
    print(f"z: {pos.z_val:.3f}")
    print(f"altitude: {-pos.z_val:.3f} m")

    lidar_data = client.getLidarData(
        lidar_name=lidar_name,
        vehicle_name=vehicle_name
    )

    points = np.array(lidar_data.point_cloud, dtype=np.float32)

    print("\n========== LiDAR Info ==========")
    print("Lidar name:", lidar_name)
    print("Timestamp:", lidar_data.time_stamp)
    print("Raw point length:", len(points))

    if len(points) == 0:
        print("No lidar data")
        return None

    points = points.reshape(-1, 3)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    distances_3d = np.linalg.norm(points, axis=1)
    distances_xy = np.sqrt(x ** 2 + y ** 2)
    angles = np.degrees(np.arctan2(y, x))

    print("\n========== Point Cloud Shape ==========")
    print("Total points:", points.shape[0])
    print("Point shape:", points.shape)
    print("First 10 points:")
    print(points[:10])

    print("\n========== Distance Stats ==========")
    print(f"3D min distance: {np.min(distances_3d):.3f}")
    print(f"3D max distance: {np.max(distances_3d):.3f}")
    print(f"XY min distance: {np.min(distances_xy):.3f}")
    print(f"XY max distance: {np.max(distances_xy):.3f}")

    print("\n========== XYZ Range ==========")
    print(f"x range: [{np.min(x):.3f}, {np.max(x):.3f}]")
    print(f"y range: [{np.min(y):.3f}, {np.max(y):.3f}]")
    print(f"z range: [{np.min(z):.3f}, {np.max(z):.3f}]")

    print("\n========== Angle Range ==========")
    print(f"horizontal angle range: [{np.min(angles):.2f}, {np.max(angles):.2f}] degrees")

    print("\nTest completed.")

    return client, points

def visualize_lidar_points_with_grid(
    points,
    max_range=15.0,
    h_sectors=36,
    vertical_fov=(-30.0, 30.0),
    v_layers=3
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    distances = np.linalg.norm(points, axis=1)
    distances_clip = np.clip(distances, 0, max_range)

    # 1. draw real lidar points
    ax.scatter(
        x, y, z,
        c=distances_clip,
        s=2,
        alpha=0.7
    )

    # 2. draw drone at origin
    ax.scatter([0], [0], [0], s=80, marker="o")
    ax.text(0, 0, 0, " Drone / LiDAR", fontsize=10)

    # 3. draw horizontal sector boundary lines
    for i in range(h_sectors):
        angle_deg = i * 360.0 / h_sectors
        angle_rad = np.radians(angle_deg)

        x_line = [0, max_range * np.cos(angle_rad)]
        y_line = [0, max_range * np.sin(angle_rad)]
        z_line = [0, 0]

        ax.plot(x_line, y_line, z_line, linewidth=0.5, alpha=0.35)

    # 4. draw vertical layer cone surfaces / rings
    v_min, v_max = vertical_fov
    v_bounds = np.linspace(v_min, v_max, v_layers + 1)

    theta = np.linspace(0, 2 * np.pi, 181)
    r = np.linspace(0, max_range, 40)
    theta_grid, r_grid = np.meshgrid(theta, r)

    for v_angle in v_bounds:
        beta = np.radians(v_angle)

        x_grid = r_grid * np.cos(theta_grid)
        y_grid = r_grid * np.sin(theta_grid)
        z_grid = r_grid * np.tan(beta)

        ax.plot_wireframe(
            x_grid,
            y_grid,
            z_grid,
            linewidth=0.3,
            alpha=0.25
        )

        # label the vertical boundary at front direction
        label_x = max_range * 0.8
        label_y = 0
        label_z = label_x * np.tan(beta)
        ax.text(label_x, label_y, label_z, f"{v_angle:.0f}°", fontsize=9)

    # 5. draw layer labels
    layer_names = ["Lower layer", "Middle layer", "Upper layer"]
    for i in range(v_layers):
        low = v_bounds[i]
        high = v_bounds[i + 1]
        mid = (low + high) / 2
        beta = np.radians(mid)

        label_x = max_range * 0.55
        label_y = max_range * 0.55
        label_z = np.sqrt(label_x ** 2 + label_y ** 2) * np.tan(beta)

        ax.text(
            label_x,
            label_y,
            label_z,
            f"{layer_names[i]}\n{low:.0f}°~{high:.0f}°",
            fontsize=9
        )

    # 6. axis labels
    ax.set_xlabel("X (forward/back)")
    ax.set_ylabel("Y (left/right)")
    ax.set_zlabel("Z (up/down)")
    ax.set_title("3D LiDAR Point Cloud with 36 × 3 Sector Grid")

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    z_limit = max_range * np.tan(np.radians(max(abs(v_min), abs(v_max)))) * 1.2
    ax.set_zlim(-z_limit, z_limit)

    ax.view_init(elev=28, azim=-55)

    plt.show()

def build_lidar_36x3_grid_from_points(
    points,
    max_range=15.0,
    h_sectors=36,
    v_layers=3,
    vertical_fov=(-30.0, 30.0)
):
    if points is None or points.shape[0] == 0:
        return np.full((v_layers, h_sectors), max_range, dtype=np.float32), []

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    distances_3d = np.linalg.norm(points, axis=1)
    xy_dist = np.sqrt(x ** 2 + y ** 2)

    horizontal_angles = np.degrees(np.arctan2(y, x))
    horizontal_angles = (horizontal_angles + 360) % 360

    vertical_angles = np.degrees(np.arctan2(z, xy_dist))

    v_min, v_max = vertical_fov

    valid_mask = (
        (distances_3d > 0.1) &
        (distances_3d <= max_range) &
        (vertical_angles >= v_min) &
        (vertical_angles <= v_max)
    )

    grid = np.full((v_layers, h_sectors), max_range, dtype=np.float32)
    selected_points = {}

    h_bin_size = 360.0 / h_sectors
    v_bin_size = (v_max - v_min) / v_layers

    for p, h_angle, v_angle, dist in zip(
        points[valid_mask],
        horizontal_angles[valid_mask],
        vertical_angles[valid_mask],
        distances_3d[valid_mask]
    ):
        h_idx = int(h_angle // h_bin_size)
        v_idx = int((v_angle - v_min) // v_bin_size)

        h_idx = min(max(h_idx, 0), h_sectors - 1)
        v_idx = min(max(v_idx, 0), v_layers - 1)

        if dist < grid[v_idx, h_idx]:
            grid[v_idx, h_idx] = dist
            selected_points[(v_idx, h_idx)] = (v_idx, h_idx, p, dist)

    return grid, list(selected_points.values())

def visualize_lidar_36x3_selected_points(
    points,
    max_range=15.0,
    h_sectors=36,
    vertical_fov=(-30.0, 30.0),
    v_layers=3
):
    grid, selected_points = build_lidar_36x3_grid_from_points(
        points,
        max_range=max_range,
        h_sectors=h_sectors,
        v_layers=v_layers,
        vertical_fov=vertical_fov
    )

    selected_xyz = []
    selected_dist = []

    for v_idx, h_idx, p, dist in selected_points:
        selected_xyz.append(p)
        selected_dist.append(dist)

    if len(selected_xyz) == 0:
        print("No selected points in 36x3 grid.")
        return

    selected_xyz = np.array(selected_xyz)
    selected_dist = np.array(selected_dist)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # raw points: light background
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=1,
        alpha=0.12,
        label="Raw LiDAR points"
    )

    # selected grid points: larger markers
    ax.scatter(
        selected_xyz[:, 0],
        selected_xyz[:, 1],
        selected_xyz[:, 2],
        c=selected_dist,
        s=45,
        alpha=1.0,
        marker="o",
        label="36x3 selected nearest points"
    )

    # drone
    ax.scatter([0], [0], [0], s=100, marker="x", label="Drone / LiDAR")

    # horizontal sector lines
    for i in range(h_sectors):
        angle_rad = np.radians(i * 360.0 / h_sectors)
        ax.plot(
            [0, max_range * np.cos(angle_rad)],
            [0, max_range * np.sin(angle_rad)],
            [0, 0],
            linewidth=0.4,
            alpha=0.25
        )

    # vertical boundaries
    v_min, v_max = vertical_fov
    v_bounds = np.linspace(v_min, v_max, v_layers + 1)

    theta = np.linspace(0, 2 * np.pi, 181)
    r = np.linspace(0, max_range, 40)
    theta_grid, r_grid = np.meshgrid(theta, r)

    for v_angle in v_bounds:
        beta = np.radians(v_angle)

        x_grid = r_grid * np.cos(theta_grid)
        y_grid = r_grid * np.sin(theta_grid)
        z_grid = r_grid * np.tan(beta)

        ax.plot_wireframe(
            x_grid,
            y_grid,
            z_grid,
            linewidth=0.25,
            alpha=0.2
        )

    ax.set_xlabel("X (forward/back)")
    ax.set_ylabel("Y (left/right)")
    ax.set_zlabel("Z (up/down)")
    ax.set_title("36×3 LiDAR Grid Selected Points vs Raw Point Cloud")

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    z_limit = max_range * np.tan(np.radians(max(abs(v_min), abs(v_max)))) * 1.2
    ax.set_zlim(-z_limit, z_limit)

    ax.legend()
    ax.view_init(elev=28, azim=-55)

    plt.show()

    print("\n========== 36x3 Grid Summary ==========")
    print("Grid shape:", grid.shape)
    print("Selected cells:", len(selected_points), "/", h_sectors * v_layers)
    print("Grid min distance:", np.min(grid))
    print("Grid max distance:", np.max(grid))
    print("Grid:")
    print(grid)

def collect_lidar_points_from_client(client, lidar_name="LidarSensor1", vehicle_name=""):
    lidar_data = client.getLidarData(
        lidar_name=lidar_name,
        vehicle_name=vehicle_name
    )

    points = np.array(lidar_data.point_cloud, dtype=np.float32)

    print("\n========== LiDAR Info ==========")
    print("Lidar name:", lidar_name)
    print("Timestamp:", lidar_data.time_stamp)
    print("Raw point length:", len(points))

    if len(points) == 0:
        print("No lidar data")
        return None

    points = points.reshape(-1, 3)

    print_lidar_stats(points)

    return points


def print_lidar_stats(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    distances_3d = np.linalg.norm(points, axis=1)
    distances_xy = np.sqrt(x ** 2 + y ** 2)
    angles = np.degrees(np.arctan2(y, x))

    print("\n========== Point Cloud Shape ==========")
    print("Total points:", points.shape[0])
    print("Point shape:", points.shape)
    print("First 10 points:")
    print(points[:10])

    print("\n========== Distance Stats ==========")
    print(f"3D min distance: {np.min(distances_3d):.3f}")
    print(f"3D max distance: {np.max(distances_3d):.3f}")
    print(f"XY min distance: {np.min(distances_xy):.3f}")
    print(f"XY max distance: {np.max(distances_xy):.3f}")

    print("\n========== XYZ Range ==========")
    print(f"x range: [{np.min(x):.3f}, {np.max(x):.3f}]")
    print(f"y range: [{np.min(y):.3f}, {np.max(y):.3f}]")
    print(f"z range: [{np.min(z):.3f}, {np.max(z):.3f}]")

    print("\n========== Angle Range ==========")
    print(f"horizontal angle range: [{np.min(angles):.2f}, {np.max(angles):.2f}] degrees")


def visualize_lidar_probe_result(
    points,
    max_range=15.0,
    h_sectors=36,
    vertical_fov=(-30.0, 30.0),
    v_layers=3
):
    visualize_lidar_points_with_grid(
        points,
        max_range=max_range,
        h_sectors=h_sectors,
        vertical_fov=vertical_fov,
        v_layers=v_layers
    )

    visualize_lidar_36x3_selected_points(
        points,
        max_range=max_range,
        h_sectors=h_sectors,
        vertical_fov=vertical_fov,
        v_layers=v_layers
    )

def compare_local_grid_with_sensor_probe(
    client,
    raw_points,
    lidar_name="LidarSensor1",
    max_range=15.0,
    h_sectors=36,
    v_layers=3,
    vertical_fov=(-30.0, 30.0)
):
    local_grid, _ = build_lidar_36x3_grid_from_points(
        raw_points,
        max_range=max_range,
        h_sectors=h_sectors,
        v_layers=v_layers,
        vertical_fov=vertical_fov
    )

    probe_grid = get_lidar_36x3_sector_distances(
        client,
        lidar_name=lidar_name,
        max_range=max_range,
        h_sectors=h_sectors,
        v_layers=v_layers,
        vertical_fov=vertical_fov
    )

    diff = np.abs(local_grid - probe_grid)

    print("\n========== Compare Local Grid vs sensor_probe Grid ==========")
    print("Local grid shape:", local_grid.shape)
    print("Probe grid shape:", probe_grid.shape)
    print("Max diff:", np.max(diff))
    print("Mean diff:", np.mean(diff))
    print("All close:", np.allclose(local_grid, probe_grid, atol=1e-5))

    print("\nProbe grid:")
    print(probe_grid)

    return probe_grid

if __name__ == "__main__":
    client, points = get_lidar_test_data(
        altitude=3.0,
        lidar_name="LidarSensor1"
    )

    if points is not None:
        visualize_lidar_points_with_grid(points)
        visualize_lidar_36x3_selected_points(points)

        compare_local_grid_with_sensor_probe(
            client,
            raw_points=points,
            lidar_name="LidarSensor1"
        )