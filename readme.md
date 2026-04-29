# 1 Environment Setup & Installation
This repository contains the algorithm and control codebase for the CS530 autonomous drone delivery project. Our simulation relies on Microsoft AirSim built on Unreal Engine.

To ensure a smooth setup and prevent version control bloat, we strictly separate the heavy 3D physics assets from our Python codebase. 

**Required Directory Structure:**
```text
CS530/
├── AirSimNH/               # (DO NOT commit) The extracted AirSim 3D environment
├── AirSimNH.zip            # (DO NOT commit) Original environment package
└── workspace/              # (Git Repository) Our Python codebase
    └── requirements.txt

## 1.1 AirSim 3D Environment Setup

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
      "Yaw": 0
    }
  },
  "ViewMode": "FlyWithMe"
}
```
- Relaunch `AirSimNH.exe`. You should see a black quadcopter parked on the street.

## 1.2 Python Development Environment
- Create Conda Environment
```bash
conda create -n cs530 python=3.9 -y
conda activate cs530
```
- Installing Dependencies
```Bash
pip install -r requirements.txt --no-build-isolation
```