import datetime
import os
import config

import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt


SCALE = 2.56
OFFSET_X = 512
OFFSET_Y = 512

OBSTACLE_GRID_PATH = r"/figures/system/obstacle_grid.png"


def pixel_to_world(px, py):
    airsim_y = (px - OFFSET_X) / SCALE
    airsim_x = -(py - OFFSET_Y) / SCALE
    return airsim_x, airsim_y


def world_to_pixel(airsim_x, airsim_y):
    px = int(airsim_y * SCALE + OFFSET_X)
    py = int(-airsim_x * SCALE + OFFSET_Y)
    return px, py


class MapSampler:
    def __init__(self, grid_path=OBSTACLE_GRID_PATH, safe_threshold=200, safety_margin_px=2):
        self.grid_path = grid_path
        self.safe_threshold = safe_threshold
        self.safety_margin_px = safety_margin_px

        self.grid = cv2.imread(grid_path, cv2.IMREAD_GRAYSCALE)
        if self.grid is None:
            raise FileNotFoundError(f"Cannot load obstacle grid: {grid_path}")

        self.height, self.width = self.grid.shape

        # White = free, black = obstacle
        free_mask = (self.grid > safe_threshold).astype(np.uint8)

        # Shrink free space by expanding obstacles
        kernel = np.ones((safety_margin_px, safety_margin_px), np.uint8)
        safe_mask = cv2.erode(free_mask, kernel, iterations=1)

        self.safe_mask = safe_mask
        self.free_pixels = np.argwhere(self.safe_mask > 0)

        if len(self.free_pixels) == 0:
            raise ValueError("No safe free pixels found in obstacle grid.")

    def _cv2_result_color(self, result):
        return config.CV2_COLORS.get(result, config.CV2_COLORS["unknown"])

    def _mpl_result_color(self, result):
        return config.MPL_COLORS.get(result, config.MPL_COLORS["unknown"])

    def init_run_dir(self, base_dir="runs", mode="train", algo="dqn", extra_info=None):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        parts = [timestamp, mode, algo]

        if extra_info:
            parts.append(extra_info)

        run_name = "_".join(parts)

        run_dir = os.path.join(base_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        self.run_dir = run_dir
        print(f"[RUN DIR] {run_dir}")

    def is_free_pixel(self, px, py):
        if px < 0 or px >= self.width or py < 0 or py >= self.height:
            return False
        return self.safe_mask[py, px] > 0

    def is_free_world(self, x, y):
        px, py = world_to_pixel(x, y)
        return self.is_free_pixel(px, py)

    def sample_free_pixel(self):
        py, px = self.free_pixels[random.randint(0, len(self.free_pixels) - 1)]
        return int(px), int(py)

    def sample_free_point(self):
        px, py = self.sample_free_pixel()
        x, y = pixel_to_world(px, py)
        return x, y

    def sample_start_goal(self, min_distance=30.0, max_attempts=1000):
        for _ in range(max_attempts):
            start = self.sample_free_point()
            goal = self.sample_free_point()

            dist = math.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)

            if dist >= min_distance:
                return start, goal

        raise RuntimeError("Failed to sample valid start/goal pair.")

    def visualize_sample(self, start, goal, window_name="Sampled Start and Goal"):
        img = cv2.cvtColor(self.grid, cv2.COLOR_GRAY2BGR)

        sx, sy = world_to_pixel(start[0], start[1])
        gx, gy = world_to_pixel(goal[0], goal[1])

        cv2.circle(img, (sx, sy), 6, (0, 255, 0), -1)
        cv2.putText(img, "Start", (sx + 8, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.circle(img, (gx, gy), 6, (0, 0, 255), -1)
        cv2.putText(img, "Goal", (gx + 8, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.line(img, (sx, sy), (gx, gy), (255, 0, 0), 1)

        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def sample_free_pixel_in_roi(self, roi):
        x_min, y_min, x_max, y_max = roi

        candidates = []
        for py, px in self.free_pixels:
            if x_min <= px <= x_max and y_min <= py <= y_max:
                candidates.append((px, py))

        if len(candidates) == 0:
            raise RuntimeError("No free pixels found in ROI.")

        return random.choice(candidates)

    def sample_free_point_in_roi(self, roi):
        px, py = self.sample_free_pixel_in_roi(roi)
        return pixel_to_world(px, py)

    def sample_start_goal_in_roi(self, roi, min_distance=20.0, max_distance=80.0, max_attempts=1000):
        for _ in range(max_attempts):
            start = self.sample_free_point_in_roi(roi)
            goal = self.sample_free_point_in_roi(roi)

            dist = math.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)

            if min_distance <= dist <= max_distance:
                return start, goal

        raise RuntimeError("Failed to sample valid start/goal pair in ROI.")

    def save_training_roi_plot(self, roi, filename="training_roi.png"):
        if not hasattr(self, "run_dir"):
            raise RuntimeError("Run directory not initialized. Call init_run_dir() first.")

        x_min, y_min, x_max, y_max = roi

        img = cv2.cvtColor(self.grid, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(
            img,
            (x_min, y_min),
            (x_max, y_max),
            (0, 0, 255),
            2
        )

        cv2.putText(
            img,
            "Training ROI",
            (x_min + 5, y_min - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        save_path = os.path.join(self.run_dir, filename)
        cv2.imwrite(save_path, img)

        print(f"[SAVE] {save_path}")

    def save_altitude_plot(
            self,
            path,
            episode,
            result="unknown",
            min_flight_altitude=None,
            max_flight_altitude=None
    ):
        if not hasattr(self, "run_dir"):
            raise RuntimeError("Run directory not initialized.")

        if len(path) == 0:
            return

        steps = list(range(len(path)))
        alts = [-p[2] for p in path]

        color = self._mpl_result_color(result)

        plt.figure(figsize=(8, 4))
        plt.plot(steps, alts, color=color, linewidth=2, label="altitude")

        if min_flight_altitude is not None:
            plt.axhline(
                y=min_flight_altitude,
                linestyle="--",
                linewidth=1.5,
                color="gray",
                label=f"min altitude = {min_flight_altitude}m"
            )

        if max_flight_altitude is not None:
            plt.axhline(
                y=max_flight_altitude,
                linestyle="--",
                linewidth=1.5,
                color="black",
                label=f"max altitude = {max_flight_altitude}m"
            )

        plt.xlabel("step")
        plt.ylabel("altitude (m)")
        plt.title(f"Altitude Curve - Ep {episode} ({result})")
        plt.grid(True)
        plt.legend()

        if min_flight_altitude is not None and max_flight_altitude is not None:
            plt.ylim(
                max(0, min_flight_altitude - 1),
                max_flight_altitude + 2
            )

        save_path = os.path.join(
            self.run_dir,
            f"altitude_ep_{episode}_{result}.png"
        )

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[SAVE] {save_path}")

    def save_trajectory_plot(self, start, goal, path, episode, roi=None, result="unknown"):
        if not hasattr(self, "run_dir"):
            raise RuntimeError("Run directory not initialized. Call init_run_dir() first.")

        img = cv2.cvtColor(self.grid, cv2.COLOR_GRAY2BGR)

        # success = green path, collision = red path, timeout/unknown = blue path
        path_color = self._cv2_result_color(result)
        start_color = config.CV2_COLORS["start"]
        goal_color = config.CV2_COLORS["goal"]
        roi_color = config.CV2_COLORS["roi"]

        # draw ROI
        if roi is not None:
            x_min, y_min, x_max, y_max = roi
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), roi_color, 1)
            cv2.putText(img, "Training ROI", (x_min + 5, y_min - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)

        sx, sy = world_to_pixel(start[0], start[1])
        gx, gy = world_to_pixel(goal[0], goal[1])

        # draw path
        if len(path) > 1:
            pts = []
            for p in path:
                x = p[0]
                y = p[1]

                px, py = world_to_pixel(x, y)
                pts.append([px, py])

            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, path_color, 2)

        # draw start
        cv2.circle(img, (sx, sy), 6, start_color, -1)
        cv2.putText(img, "Start", (sx + 8, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, start_color, 1)

        # draw goal
        cv2.circle(img, (gx, gy), 6, goal_color, -1)
        cv2.putText(img, "Goal", (gx + 8, gy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, goal_color, 1)

        # result label
        cv2.putText(img, f"Episode {episode}: {result}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, path_color, 2)

        save_path = os.path.join(self.run_dir, f"trajectory_ep_{episode}_{result}.png")

        cv2.imwrite(save_path, img)
        print(f"[SAVE] {save_path}")

    def save_trajectory_3d_plot(
            self,
            start,
            goal,
            path,
            episode,
            result="unknown",
            min_flight_altitude=None,
            max_flight_altitude=None
    ):
        if not hasattr(self, "run_dir"):
            raise RuntimeError("Run directory not initialized. Call init_run_dir() first.")

        if len(path) == 0:
            return

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        alts = [-p[2] for p in path]

        start_alt = -start[2]
        goal_alt = -goal[2]

        path_color = self._mpl_result_color(result)
        start_color = config.MPL_COLORS["start"]
        goal_color = config.MPL_COLORS["goal"]

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            xs,
            ys,
            alts,
            linewidth=3,
            color=path_color,
            label="drone path"
        )

        ax.scatter(
            start[0],
            start[1],
            start_alt,
            s=80,
            color=start_color,
            label="start"
        )

        ax.scatter(
            goal[0],
            goal[1],
            goal_alt,
            s=80,
            color=goal_color,
            label="goal"
        )

        ax.scatter(
            xs[-1],
            ys[-1],
            alts[-1],
            s=70,
            marker="x",
            color=path_color,
            label="final"
        )

        # =========================
        # Altitude constraint planes
        # =========================
        if min_flight_altitude is not None and max_flight_altitude is not None:
            x_min = min(xs + [start[0], goal[0]])
            x_max = max(xs + [start[0], goal[0]])
            y_min = min(ys + [start[1], goal[1]])
            y_max = max(ys + [start[1], goal[1]])

            # 防止 path 太短导致平面太窄
            if abs(x_max - x_min) < 1e-6:
                x_min -= 1
                x_max += 1

            if abs(y_max - y_min) < 1e-6:
                y_min -= 1
                y_max += 1

            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 12),
                np.linspace(y_min, y_max, 12)
            )

            zz_min = np.full_like(xx, min_flight_altitude)
            zz_max = np.full_like(xx, max_flight_altitude)

            ax.plot_surface(
                xx,
                yy,
                zz_min,
                alpha=0.15,
                color="gray",
                edgecolor="none"
            )

            ax.plot_surface(
                xx,
                yy,
                zz_max,
                alpha=0.12,
                color="black",
                edgecolor="none"
            )

            ax.text(
                x_min,
                y_min,
                min_flight_altitude,
                f"min altitude = {min_flight_altitude}m",
                color="gray"
            )

            ax.text(
                x_min,
                y_min,
                max_flight_altitude,
                f"max altitude = {max_flight_altitude}m",
                color="black"
            )

            ax.set_zlim(
                max(0, min_flight_altitude - 1),
                max_flight_altitude + 2
            )

        else:
            max_alt = max(alts + [start_alt, goal_alt])
            min_alt = min(alts + [start_alt, goal_alt])
            ax.set_zlim(max(0, min_alt - 1), max_alt + 1)

        ax.set_title(f"Episode {episode}: {result}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("altitude (m)")

        ax.view_init(elev=25, azim=-60)
        ax.legend()

        save_path = os.path.join(
            self.run_dir,
            f"trajectory3d_ep_{episode}_{result}.png"
        )

        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[SAVE] {save_path}")

if __name__ == "__main__":
    sampler = MapSampler()

    # === 1. 初始化保存目录 ===
    sampler.init_run_dir(name="roi_test")

    # === 2. 定义训练区域（你的红框）===
    training_roi = (80, 500, 500, 980)

    # === 3. 保存 ROI 图 ===
    sampler.save_training_roi_plot(training_roi)

    # === 4. 多采样几组 start/goal 测试 ===
    for i in range(1):
        start, goal = sampler.sample_start_goal_in_roi(
            roi=training_roi,
            min_distance=20.0,
            max_distance=80.0
        )

        print(f"\nSample {i+1}")
        print("Start:", start)
        print("Goal:", goal)

        sampler.visualize_sample(start, goal)