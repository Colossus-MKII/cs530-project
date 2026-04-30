import airsim
import numpy as np
import cv2

client = airsim.CarClient()
client.confirmConnection()

responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
    airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
])

# ===== Scene 图像 =====
scene = responses[0]
img_rgb = np.frombuffer(scene.image_data_uint8, dtype=np.uint8)
img_rgb = img_rgb.reshape(scene.height, scene.width, 3)

# ===== Depth 图像 =====
depth = responses[1]
img_depth = np.array(depth.image_data_float, dtype=np.float32)
img_depth = img_depth.reshape(depth.height, depth.width)

# 归一化显示（否则全黑）
img_depth_vis = np.clip(img_depth, 0, 20)
img_depth_vis = (img_depth_vis / 20.0 * 255).astype(np.uint8)

cv2.imshow("Scene", img_rgb)
cv2.imshow("Depth", img_depth_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()