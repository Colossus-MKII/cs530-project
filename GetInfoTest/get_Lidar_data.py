import airsim
import numpy as np

client = airsim.CarClient()
client.confirmConnection()

lidar_data = client.getLidarData("LidarSensor1")

points = np.array(lidar_data.point_cloud, dtype=np.float32)

if len(points) == 0:
    print("No lidar data")
else:
    points = points.reshape(-1, 3)

    print("Total points:", points.shape[0])
    print("First 5 points:\n", points[:5])

    # 计算距离
    distances = np.linalg.norm(points, axis=1)

    print("Min distance:", np.min(distances))
    print("Max distance:", np.max(distances))