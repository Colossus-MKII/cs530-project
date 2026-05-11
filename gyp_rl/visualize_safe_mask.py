import cv2
import numpy as np

GRID_PATH = "..\obstacle_grid.png"

def create_circular_kernel(radius):
    size = radius * 2 + 1
    kernel = np.zeros((size, size), np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    return kernel


def visualize_safety_margin(safety_margin_px=20):
    grid = cv2.imread(GRID_PATH, cv2.IMREAD_GRAYSCALE)
    if grid is None:
        raise FileNotFoundError(f"Cannot load {GRID_PATH}")

    # 原始 free mask
    free_mask = (grid > 200).astype(np.uint8)

    # 膨胀障碍（=收缩 free space）
    kernel = create_circular_kernel(radius=8)
    safe_mask = cv2.erode(free_mask, kernel, iterations=1)

    # 可视化
    original_vis = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)

    # 红色 = 被“吃掉”的区域（原来安全，现在不安全）
    removed = (free_mask == 1) & (safe_mask == 0)

    vis = original_vis.copy()
    vis[removed] = (0, 0, 255)   # 红色标出危险边界

    safe_vis = (safe_mask * 255).astype(np.uint8)

    cv2.imshow("Original Grid", original_vis)
    cv2.imshow("Safe Mask (After Erosion)", safe_vis)
    cv2.imshow(f"Overlay (margin={safety_margin_px})", vis)

    print(f"Safety margin: {safety_margin_px}px")
    print("White = safe, Black = obstacle, Red = removed unsafe boundary")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for margin in [1,2,3]:
        visualize_safety_margin(margin)