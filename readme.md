# How to build up the environment and run the program

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

- Run `AirSimNH.exe`, hanging up the application
- Run `telemetry_map.py` to visualize the drone's telemetry data on a global map
    - ```Bash
    - cd workspace
    - python telemetry_map.py```
    - Then you will see a window named `Global Delivery Map`, it indicates the path of drone
- Run `autonomous_delivery.py` to execute the hybrid navigation algorithm
    - ```Bash
    - python autonomous_delivery.py```
    - Then you will see a window named `Hybrid Navigator`
    - Click on the map to set the start (Green) and end (Red) points for delivery. The drone will autonomously navigate while avoiding obstacles.
    - After planning the path, you will see 3 choices:
        - Press `s` to execute the planned path in AirSim
        - Press `q` to quit the program
    - After the planned path executed, you can press `r` to reset the start and end and run again.