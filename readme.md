# How to build up the environment and run the program

**Project Video Part1 (A* + APF)**

[![Autonomous Drone Navigation Demo](https://img.youtube.com/vi/a9YAp_fo9fU/maxresdefault.jpg)](https://www.youtube.com/watch?v=a9YAp_fo9fU)

---

**Project Video Part2 (Train DQN)**

[![Autonomous Drone Navigation Demo](https://img.youtube.com/vi/LNvynDGoq9k/maxresdefault.jpg)](https://www.youtube.com/watch?v=LNvynDGoq9k)

---

# 1 Environment Setup & Installation

This repository contains the algorithm and control codebase for the CS530 autonomous drone delivery project. Our simulation relies on Microsoft AirSim built on Unreal Engine.

To ensure a smooth setup and prevent version control bloat, we strictly separate the heavy 3D physics assets from our Python codebase.

---

## Required Directory Structure

```text
CS530/
├── AirSimNH/               # (DO NOT commit) The extracted AirSim 3D environment
├── AirSimNH.zip            # (DO NOT commit) Original environment package
└── workspace/              # (Git Repository) Our Python codebase
    └── requirements.txt
```

---

## 1.1 AirSim 3D Environment Setup

- Download the latest AirSim package `AirSimNH.zip`
- Extract the environment
- Configure AirSim drone settings

Navigate to:

```text
C:\Users\<YourUsername>\Documents\AirSim\
```

Edit `settings.json`:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1.0,
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "X": 0,
      "Y": 0,
      "Z": 0,
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
          "X": 0,
          "Y": 0,
          "Z": 0,
          "Yaw": 0,
          "Pitch": -90,
          "Roll": 0,
          "DrawDebugPoints": false
        }
      }
    }
  },
  "ViewMode": "FlyWithMe"
}
```

Relaunch `AirSimNH.exe`.

You should see a black quadcopter placed on the street.

---

## 1.2 Python Development Environment

### Create Conda Environment

```bash
conda create -n cs530 python=3.9 -y
conda activate cs530
```

### Install Dependencies

```bash
pip install -r requirements.txt --no-build-isolation
```

---

# 2 How to use the navigation system?

This part uses A* path planning and Artificial Potential Field (APF) to avoid obstacle collisions.

---

## Run telemetry visualization

```bash
python telemetry_map.py
```

A window named `Global Delivery Map` will appear to visualize the drone trajectory.

---

## Run autonomous delivery system

```bash
python autonomous_delivery.py
```

A window named `Hybrid Navigator` will appear.

### Controls

- Click to select:
  - Start point (Green)
  - Goal point (Red)

### Keyboard Commands

- `s` → execute planned path
- `r` → reset and restart
- `q` → quit

---

# 3 Reinforcement Learning (DQN Navigation System)

This project also includes a reinforcement learning based drone navigation framework using Deep Q-Network (DQN).

The RL system is implemented under:

```text
gyp_rl/
```

---

# 3.1 RL Project Structure

```text
gyp_rl/
├── analysis_results/          # training analysis and experiment outputs
├── runs/                      # saved training runs and logs
├── airsim_env.py              # AirSim RL environment
├── analysis.py                # experiment analysis
├── config.py                  # training configuration
├── map_sampler.py             # random map / sampling logic
├── Reward_states.md           # detailed reward design explanation
├── RL-training-plan.md        # complete RL training roadmap
├── sensor_probe.py            # sensor debugging tools
├── test_reward.py             # reward testing utilities
├── train_dqn.py               # DQN training entry point
├── visualize_safe_mask.py     # visualization utilities
└── __init__.py
```

---

# 3.2 RL Training Pipeline

The RL framework follows a staged curriculum-learning strategy:

1. Train in soft environments
2. Train in hard environments
3. Improve robustness under dense obstacles
4. Future work:
   - Multi-altitude training
   - CNN-based visual navigation
   - Cross-environment generalization

The current implementation uses:

- DQN
- Lidar sectorization
- Discrete action space
- Reward shaping
- Safety constraints

---

# 3.3 Reward Function Design

Detailed reward design and safety constraint logic are documented in:

```text
gyp_rl/Reward_states.md
```

This document explains:

- Soft vs hard environment reward differences
- Progress reward
- Obstacle penalties
- Altitude constraints
- Terminal conditions
- Reward philosophy

---

# 3.4 RL Training Roadmap

The complete RL development and experiment roadmap is documented in:

```text
gyp_rl/RL-training-plan.md
```

This includes:

- Environment design
- Sensor configuration
- State representation
- Action space
- Multi-stage training strategy
- CNN future work
- Multi-altitude training
- Generalization testing
- Noise robustness experiments

---

# 3.5 Run DQN Training

Start RL training:

```bash
python gyp_rl/train_dqn.py
```