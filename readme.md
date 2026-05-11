# How to build up the environment and run the program
**Video Demo**
[![Autonomous Drone Navigation Demo](https://img.youtube.com/vi/a9YAp_fo9fU/0.jpg)](https://www.youtube.com/watch?v=a9YAp_fo9fU)



## 1 Environment Setup & Installation
This repository contains the algorithm and control codebase for the CS530 autonomous drone delivery project. Our simulation relies on Microsoft AirSim built on Unreal Engine.

To ensure a smooth setup and prevent version control bloat, we strictly separate the heavy 3D physics assets from our Python codebase. 

**Required Directory Structure:**
```text
CS530/
├── AirSimNH/               # (DO NOT commit) The extracted AirSim 3D environment
├── AirSimNH.zip            # (DO NOT commit) Original environment package
└── workspace/              # (Git Repository) Our Python codebase
    └── requirements.txt
```
### 1.1 AirSim 3D Environment Setup

- Download the latest AirSim package `AirSimNH.zip` from https://github.com/Microsoft/AirSim/releases and Extract the Environment
- Configure the Drone (settings.json)
By default, AirSim spawns a car. We need to configure it to spawn a quadcopter drone at the absolute origin (0,0,0).
- Navigate to `C:\Users\<YourUsername>\Documents\AirSim\` 
- Edit the `settings.json` file as following:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1.0,
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": 0,
      "Yaw": 0,
      "Sensors": {
        "LidarSensor1": {
          "SensorType": 6,
          "Enabled": true,
          "NumberOfChannels": 16,
          "RotationsPerSecond": 10,
          "PointsPerSecond": 10000,
          "VerticalFOVUpper": 7,
          "VerticalFOVLower": -7,
          "HorizontalFOVStart": -180,
          "HorizontalFOVEnd": 180,
          "DrawDebugPoints": true,
          "DataFrame": "SensorLocalFrame"
        },
        "DistanceSensor_Down": {
          "SensorType": 5,
          "Enabled": true,
          "X": 0, "Y": 0, "Z": 0,
          "Yaw": 0, "Pitch": -90, "Roll": 0,
          "DrawDebugPoints": false
        }
      }
    }
  },
  "ViewMode": "FlyWithMe"
}
```
- Relaunch `AirSimNH.exe`. You should see a black quadcopter parked on the street.

### 1.2 Python Development Environment
- Create Conda Environment
```bash
conda create -n cs530 python=3.9 -y
conda activate cs530
```
- Installing Dependencies
```Bash
pip install -r requirements.txt --no-build-isolation
```

## 2 How to use the navigation system?

This part uses A* path planning and Artificial Potential Field to avoid colliding with obstacle.

- Run `AirSimNH.exe`, hanging up the application
- Run `telemetry_map.py` to visualize the drone's telemetry data on a global map
```Bash
cd workspace
python telemetry_map.py
```
Then you will see a window named `Global Delivery Map`, it indicates the path of drone.
- Run `autonomous_delivery.py` to execute the hybrid navigation algorithm
```Bash
python autonomous_delivery.py
```

Then you will see a window named `Hybrid Navigator`.
   - Click on the map to set the start (Green) and end (Red) points for delivery. The drone will autonomously navigate while avoiding obstacles.
   - After planning the path, you will see 3 choices:
      - Press `s` to execute the planned path in AirSim
      - Press `q` to quit the program
      - After the planned path executed, you can press `r` to reset the start and end and run again.

## 3 Reinforcement Learning (RL) Training Plan (Work in Progress)
Note: The RL component is currently under development. Below is the structured plan for training the agent.

### 3.1 Problem Definition

We formulate the task as **fixed-altitude (≈3m) local navigation with obstacle avoidance**.
 Takeoff and landing are handled by rule-based control.
 The RL agent controls only **horizontal movement** and aims to reach a target while avoiding obstacles.

------

### 3.2 Sensor Usage

**Primary: Lidar**

- Convert point cloud into **18 angular sectors** (10° each over 180° FOV)
- For each sector, take the **minimum distance**
- **Clip** distances to 15m and **normalize** to [0, 1]

Output:
 `lidar_state = [d1, d2, ..., d18]`

**Secondary: DistanceFront**

- Included as an additional feature
- Used for **safety override** (e.g., emergency stop if too close)

**DepthPlanar**

- Used only for **visualization/debugging**
- Not included in RL input

------

### 3.3 State Representation

```
state = [
  goal_angle,
  distance_to_goal,
  lidar_sector_1,
  ...,
  lidar_sector_18,
  distance_front
]
```

- Total dimension ≈ 21
- Normalization:
  - `goal_angle / π`
  - `distance_to_goal / max_range`
  - `lidar / 15`
  - `distance_front / 15`

------

### 3.4 Action Space


Discrete actions (7 total):

- forward
- forward-left
- forward-right
- left
- right
- back
- hover

Each action is executed for a fixed duration (e.g., 0.5 seconds).

------

### 3.5 Reward Function

```
reward = progress + goal + collision + danger + step
```

**Progress**

- `K * (previous_distance - current_distance)`
- Recommended: `K ≈ 5`

**Goal**

- If `distance_to_goal < 1.0m`: +100 and terminate episode

**Collision**

- If collision occurs during flight: -100 and terminate
- Ground contact is ignored

**Danger (proximity to obstacles)**

- If `min_lidar < 0.8m`: -20
- If `min_lidar < 1.5m`: -5

**Step penalty**

- Constant penalty: -0.05 per step

------

### 3.6 RL Algorithm

Use **Deep Q-Network (DQN)**

- Input: state vector (~21D)
- Output: Q-values for each action

Network structure:

- Linear(21, 128) → ReLU
- Linear(128, 128) → ReLU
- Linear(128, action_dim)

------

### 3.7 Episode Design

**Reset**

- Reset environment
- Take off
- Move to fixed altitude (z = -3m)
- Initialize start and goal positions
- Read initial state

**Step**

- Select action (ε-greedy)
- Execute action
- Read sensors
- Compute next state and reward
- Check termination

**Termination conditions**

- Goal reached
- Collision during flight
- Max steps exceeded (e.g., 200)

------

### 3.8 Training Strategy

- Stage 1: Empty environment (learn to move toward goal)
- Stage 2: Add simple obstacles (learn avoidance)
- Stage 3: Evaluate robustness under noise

------

### 3.9 Noise Experiment

Add Gaussian noise to Lidar inputs:

```
d_noisy = d + N(0, σ)
```

Test values:

- σ = 0, 0.1, 0.3, 0.5

Metrics:

- Success rate
- Collision rate
- Path length
- Average reward

------

### 3.10 Key Design Choices

- Fix altitude to reduce the problem to 2D
- Use Lidar sectorization to avoid high-dimensional input
- Use DQN to handle continuous state and discrete actions
- Add a safety override using DistanceFront
- Ignore ground collision to prevent reward bias
