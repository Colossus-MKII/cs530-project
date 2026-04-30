import airsim
import numpy as np

client = airsim.CarClient()
client.confirmConnection()

# 获取 DepthPlanar 图像
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, pixels_as_float=True)
])

response = responses[0]

# 转 numpy
depth = np.array(response.image_data_float, dtype=np.float32)
depth = depth.reshape(response.height, response.width)

print("Shape:", depth.shape)
print("Min depth:", np.min(depth))
print("Max depth:", np.max(depth))
print("Center pixel depth:", depth[depth.shape[0]//2, depth.shape[1]//2])