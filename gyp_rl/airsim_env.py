import time
import math
import numpy as np
import airsim
import cv2

from map_sampler import MapSampler, world_to_pixel
from sensor_probe import (
    connect_drone,
    get_drone_state,
    get_collision_info,
    get_lidar_36x3_sector_distances,
    get_distance_sensor_data,
)


class AirSimDroneEnv:
    def __init__(
        self,
        goal=(20.0, 0.0, -3.0),
        init_altitude=-3.0,
        max_steps=200,
        max_lidar_range=15.0,
        max_goal_distance=80.0,
        action_duration=0.2,
        velocity=4.0,
    ):
        self.client = connect_drone(enable_api_control=True, arm=True)

        self.init_altitude = float(init_altitude)
        self.min_flight_altitude = 2.0
        self.max_flight_altitude = 20.0

        self.min_altitude = -self.max_flight_altitude  # 最高 20m
        self.max_altitude = -self.min_flight_altitude  # 最低 2m

        self.goal = np.array(goal, dtype=np.float32)
        self.start = np.array((0.0, 0.0, self.init_altitude), dtype=np.float32)

        self.max_steps = max_steps
        self.max_lidar_range = max_lidar_range
        self.max_goal_distance = max_goal_distance

        self.action_duration = action_duration
        self.velocity = velocity

        self.step_count = 0
        self.prev_distance_to_goal = None

        self.action_dim = 9

        # 3D LiDAR grid
        self.lidar_h_sectors = 36
        self.lidar_v_layers = 3
        self.lidar_vertical_fov = (-30.0, 30.0)

        # goal horizontal angle, goal vertical angle, 3D goal distance, altitude
        # + lidar 36 x 3
        # + DistanceFront, DistanceUp, DistanceDown
        self.state_dim = 4 + self.lidar_h_sectors * self.lidar_v_layers + 3

        self.min_altitude = -20.0  # highest 20m
        self.max_altitude = -2.0   # lowest 2m

        self.debug_action = False

        self.sampler = MapSampler(safety_margin_px=3)
        self.show_reset_map = True
        self.training_roi = (80, 500, 500, 980)

    # =========================
    # Environment API
    # =========================

    def reset(self, start=None, goal=None, mode="train"):
        if mode == "train":
            return self.reset_train(start, goal)
        elif mode == "demo":
            return self.reset_demo(start, goal)
        else:
            raise ValueError(f"Unknown reset mode: {mode}")

    def reset_train(self, start=None, goal=None):
        self.client.reset()
        time.sleep(1.0)

        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        if start is None or goal is None:
            start_xy, goal_xy = self.sampler.sample_start_goal_in_roi(
                roi=self.training_roi,
                min_distance=20.0,
                max_distance=80.0
            )

            start = (start_xy[0], start_xy[1], self.init_altitude)
            goal = (goal_xy[0], goal_xy[1], self.init_altitude)

        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)

        print(f"[RESET-TRAIN] start={self.start}, goal={self.goal}")
        self._show_start_goal_map(self.start, self.goal)

        self.client.takeoffAsync().join()

        pose = airsim.Pose(
            airsim.Vector3r(
                float(self.start[0]),
                float(self.start[1]),
                float(self.start[2])
            ),
            airsim.to_quaternion(0, 0, 0)
        )

        self.client.simSetVehiclePose(pose, ignore_collision=True)
        time.sleep(0.5)

        self.client.moveByVelocityAsync(0.0, 0.0, 0.0, 0.3).join()
        time.sleep(0.3)

        cleared = self._wait_until_collision_clear(timeout=5.0)
        print("Collision cleared:", cleared)

        self.step_count = 0
        self.prev_distance_to_goal = self._distance_to_goal()

        print("Initial position:", get_drone_state(self.client)["position"])

        return self.get_state()

    def reset_demo(self, start=None, goal=None):
        self.client.reset()
        time.sleep(1.0)

        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        if start is None or goal is None:
            start_xy, goal_xy = self.sampler.sample_start_goal(min_distance=30.0)

            start = (start_xy[0], start_xy[1], self.init_altitude)
            goal = (goal_xy[0], goal_xy[1], self.init_altitude)

        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)

        print(f"[RESET-DEMO] start={self.start}, goal={self.goal}")
        self._show_start_goal_map(self.start, self.goal)

        pose = airsim.Pose(
            airsim.Vector3r(
                float(self.start[0]),
                float(self.start[1]),
                -0.2
            ),
            airsim.to_quaternion(0, 0, 0)
        )

        self.client.simSetVehiclePose(pose, ignore_collision=True)
        time.sleep(0.5)

        print("Taking off...")
        self.client.takeoffAsync().join()

        print("Moving to cruise altitude...")
        self.client.moveToPositionAsync(
            float(self.start[0]),
            float(self.start[1]),
            float(self.start[2]),
            2
        ).join()

        time.sleep(1.0)

        cleared = self._wait_until_collision_clear(timeout=5.0)
        print("Collision cleared:", cleared)

        self.step_count = 0
        self.prev_distance_to_goal = self._distance_to_goal()

        print("Initial position:", get_drone_state(self.client)["position"])

        return self.get_state()

    def step(self, action):
        self.step_count += 1

        distance_front = self._get_distance_raw("DistanceFront")
        distance_up = self._get_distance_raw("DistanceUp")
        distance_down = self._get_distance_raw("DistanceDown")

        if distance_front < 1.0:
            action = 6

        if action == 7 and distance_up < 1.0:
            action = 6

        if action == 8 and distance_down < 1.0:
            action = 6

        self._execute_action(action)
        time.sleep(0.1)

        state = self.get_state()
        reward, done, info = self.compute_reward(action)

        info["distance_front"] = distance_front
        info["distance_up"] = distance_up
        info["distance_down"] = distance_down

        if self.step_count >= self.max_steps:
            done = True
            info["timeout"] = True

        return state, reward, done, info

    # =========================
    # State
    # =========================

    def get_state(self):
        x, y, z = get_drone_state(self.client)["position"]

        dx = self.goal[0] - x
        dy = self.goal[1] - y
        dz = self.goal[2] - z

        horizontal_distance = math.sqrt(dx ** 2 + dy ** 2)
        distance_3d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        horizontal_angle = math.atan2(dy, dx)
        vertical_angle = math.atan2(dz, horizontal_distance + 1e-6)

        altitude = -z

        horizontal_angle_norm = horizontal_angle / math.pi
        vertical_angle_norm = vertical_angle / (math.pi / 2)
        distance_norm = min(distance_3d, self.max_goal_distance) / self.max_goal_distance
        altitude_norm = (
                                altitude - self.min_flight_altitude
                        ) / (
                                self.max_flight_altitude - self.min_flight_altitude
                        )

        altitude_norm = np.clip(altitude_norm, 0.0, 1.0)

        lidar_grid = get_lidar_36x3_sector_distances(
            self.client,
            max_range=self.max_lidar_range,
            h_sectors=self.lidar_h_sectors,
            v_layers=self.lidar_v_layers,
            vertical_fov=self.lidar_vertical_fov
        )

        lidar_grid = np.clip(
            lidar_grid,
            0.0,
            self.max_lidar_range
        ) / self.max_lidar_range

        distance_front = self._get_distance_sensor_norm("DistanceFront")
        distance_up = self._get_distance_sensor_norm("DistanceUp")
        distance_down = self._get_distance_sensor_norm("DistanceDown")

        state = np.concatenate([
            np.array([
                horizontal_angle_norm,
                vertical_angle_norm,
                distance_norm,
                altitude_norm
            ], dtype=np.float32),

            lidar_grid.flatten().astype(np.float32),

            np.array([
                distance_front,
                distance_up,
                distance_down
            ], dtype=np.float32)
        ])

        return state.astype(np.float32)

    # =========================
    # Reward
    # =========================

    def compute_reward(self, action):
        reward = 0.0
        done = False
        info = {}
        reward_terms = {}

        # =========================
        # 1. 3D progress reward
        # =========================
        current_distance = self._distance_to_goal()
        progress = self.prev_distance_to_goal - current_distance

        progress_reward = 5.0 * progress
        step_penalty = -0.2

        reward += progress_reward
        reward += step_penalty

        reward_terms["progress_reward"] = progress_reward
        reward_terms["step_penalty"] = step_penalty

        self.prev_distance_to_goal = current_distance

        # =========================
        # 2. Obstacle safety penalty
        # =========================
        min_lidar = self._min_lidar_distance()
        info["min_lidar"] = min_lidar
        info["action"] = action

        if min_lidar < 0.8:
            obstacle_penalty = -20.0
            info["too_close"] = True
        elif min_lidar < 1.5:
            obstacle_penalty = -5.0
            info["near_obstacle"] = True
        elif min_lidar < 2.5:
            obstacle_penalty = -1.0
            info["slightly_near_obstacle"] = True
        else:
            obstacle_penalty = 0.0

        reward += obstacle_penalty
        reward_terms["obstacle_penalty"] = obstacle_penalty

        # =========================
        # 3. Goal reward
        # =========================
        if current_distance < 1.0:
            goal_reward = 100.0
            reward += goal_reward
            reward_terms["goal_reward"] = goal_reward
            done = True
            info["goal_reached"] = True
        else:
            reward_terms["goal_reward"] = 0.0

        # =========================
        # 4. Altitude cost / boundary
        # =========================
        z = get_drone_state(self.client)["position"][2]
        altitude = -z

        info["z"] = z
        info["altitude"] = altitude

        altitude_energy_cost = -0.02 * altitude
        reward += altitude_energy_cost
        reward_terms["altitude_energy_cost"] = altitude_energy_cost

        if altitude > self.max_flight_altitude:
            excess_altitude = altitude - self.max_flight_altitude
            too_high_penalty = -20.0 - 2.0 * excess_altitude

            reward += too_high_penalty
            reward_terms["too_high_penalty"] = too_high_penalty
            info["too_high"] = True
        else:
            reward_terms["too_high_penalty"] = 0.0

        if altitude < self.min_flight_altitude:
            too_low_penalty = -50.0
            reward += too_low_penalty
            reward_terms["too_low_penalty"] = too_low_penalty
            done = True
            info["too_low"] = True
        else:
            reward_terms["too_low_penalty"] = 0.0

        # =========================
        # 5. ROI boundary
        # =========================
        if self._is_out_of_training_roi():
            roi_penalty = -20 - 1.5 * current_distance
            reward += roi_penalty
            reward_terms["out_of_roi_penalty"] = roi_penalty
            info["out_of_roi"] = True
        else:
            reward_terms["out_of_roi_penalty"] = 0.0

        # =========================
        # 6. Collision
        # =========================
        real_collision, collision_debug = self._is_real_collision()
        info["collision_debug"] = collision_debug

        if real_collision:
            collision_penalty = -100.0
            reward += collision_penalty
            reward_terms["collision_penalty"] = collision_penalty
            done = True
            info["collision"] = True
        else:
            reward_terms["collision_penalty"] = 0.0

        # =========================
        # 7. Debug info
        # =========================
        info["reward_terms"] = reward_terms
        info["reward_total"] = reward
        info["current_distance"] = current_distance
        info["progress"] = progress

        return reward, done, info

    # =========================
    # Action execution
    # =========================

    def _execute_action(self, action):
        vx, vy, vz = 0.0, 0.0, 0.0
        speed = self.velocity
        vertical_speed = 1.0

        if action == 0:
            vx = speed
        elif action == 1:
            vx = speed * 0.7
            vy = -speed * 0.7
        elif action == 2:
            vx = speed * 0.7
            vy = speed * 0.7
        elif action == 3:
            vy = -speed
        elif action == 4:
            vy = speed
        elif action == 5:
            vx = -speed
        elif action == 6:
            vx, vy, vz = 0.0, 0.0, 0.0
        elif action == 7:
            vz = -vertical_speed
        elif action == 8:
            vz = vertical_speed
        else:
            raise ValueError(f"Invalid action: {action}")

        current_z = get_drone_state(self.client)["position"][2]

        if action == 7 and current_z <= self.min_altitude:
            vz = 0.0

        if action == 8 and current_z >= self.max_altitude:
            vz = 0.0

        if self.debug_action:
            print(
                f"[DEBUG] action={action}, "
                f"vx={vx}, vy={vy}, vz={vz}, current_z={current_z}"
            )

        self.client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            self.action_duration
        ).join()

    # =========================
    # Position helpers
    # =========================

    def get_position_xy(self):
        x, y, _ = get_drone_state(self.client)["position"]
        return float(x), float(y)

    def get_position_xyz(self):
        x, y, z = get_drone_state(self.client)["position"]
        return float(x), float(y), float(z)

    # =========================
    # Helper functions
    # =========================

    def _distance_to_goal(self):
        x, y, z = get_drone_state(self.client)["position"]

        dx = self.goal[0] - x
        dy = self.goal[1] - y
        dz = self.goal[2] - z

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _get_distance_raw(self, sensor_name):
        data = get_distance_sensor_data(self.client, sensor_name)

        if data is None or not data.get("available", False):
            return self.max_lidar_range

        return float(data["distance"])

    def _get_distance_sensor_norm(self, sensor_name):
        raw = self._get_distance_raw(sensor_name)
        return min(raw, self.max_lidar_range) / self.max_lidar_range

    def _min_lidar_distance(self):
        lidar_grid = get_lidar_36x3_sector_distances(
            self.client,
            max_range=self.max_lidar_range,
            h_sectors=self.lidar_h_sectors,
            v_layers=self.lidar_v_layers,
            vertical_fov=self.lidar_vertical_fov
        )
        return float(np.min(lidar_grid))

    def _wait_until_collision_clear(self, timeout=5.0):
        start = time.time()

        while time.time() - start < timeout:
            collision = get_collision_info(self.client)

            if not collision["has_collided"]:
                return True

            time.sleep(0.2)

        return False

    def _is_real_collision(self):
        collision = get_collision_info(self.client)
        z = get_drone_state(self.client)["position"][2]

        has_collided = collision["has_collided"]

        debug = {
            "has_collided": has_collided,
            "z": z,
            "reason": None,
            "real_collision": False
        }

        if not has_collided:
            debug["reason"] = "no_collision_flag"
            return False, debug

        if z > -1.0:
            debug["reason"] = "ignored_near_ground"
            return False, debug

        debug["reason"] = "real_collision_after_takeoff"
        debug["real_collision"] = True

        return True, debug

    def close(self):
        try:
            print("Landing drone...")

            self.client.hoverAsync().join()
            self.client.landAsync().join()

            self.client.armDisarm(False)
            self.client.enableApiControl(False)

        except Exception as e:
            print("Close failed:", e)

    # =========================
    # Map helpers
    # =========================

    def _show_start_goal_map(self, start, goal):
        if not self.show_reset_map:
            return

        grid = cv2.imread(
            r"/figures/system/obstacle_grid.png",
            cv2.IMREAD_GRAYSCALE
        )

        if grid is None:
            print("[MAP] obstacle_grid.png not found.")
            return

        img = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)

        sx, sy = world_to_pixel(start[0], start[1])
        gx, gy = world_to_pixel(goal[0], goal[1])

        cv2.circle(img, (sx, sy), 6, (0, 255, 0), -1)
        cv2.putText(img, "Start", (sx + 8, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.circle(img, (gx, gy), 6, (0, 0, 255), -1)
        cv2.putText(img, "Goal", (gx + 8, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.line(img, (sx, sy), (gx, gy), (255, 0, 0), 2)

        cv2.imshow("Start-Goal Map", img)
        cv2.waitKey(500)

    def _is_out_of_training_roi(self):
        x, y, _ = get_drone_state(self.client)["position"]

        px, py = world_to_pixel(x, y)
        x_min, y_min, x_max, y_max = self.training_roi

        return not (x_min <= px <= x_max and y_min <= py <= y_max)


if __name__ == "__main__":
    env = AirSimDroneEnv(goal=(20, 0, -3))

    try:
        for episode in range(2):
            print(f"\n===== Episode {episode + 1} =====")

            state = env.reset(mode="train")

            print("Initial position:", get_drone_state(env.client)["position"])
            print("State shape:", state.shape)
            print("Expected state_dim:", env.state_dim)
            print("Goal features:", state[:4])
            print("LiDAR part shape:", state[4:4 + env.lidar_h_sectors * env.lidar_v_layers].shape)
            print("Distance sensors:", state[-3:])

            total_reward = 0.0

            for i in range(10):
                action = np.random.randint(env.action_dim)
                print("Action:", action)

                next_state, reward, done, info = env.step(action)
                total_reward += reward

                print(f"\nStep {i + 1}")
                print("Reward:", reward)
                print("Total reward:", total_reward)
                print("Done:", done)
                print("Info:", info)
                print("State shape:", next_state.shape)

                if done:
                    break

    finally:
        env.close()