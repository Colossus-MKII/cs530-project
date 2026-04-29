### 1. Problem Definition

We formulate the task as **fixed-altitude (≈3m) local navigation with obstacle avoidance**.
 Takeoff and landing are handled by rule-based control.
 The RL agent controls only **horizontal movement** and aims to reach a target while avoiding obstacles.

------

### 2. Sensor Usage

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

### 3. State Representation

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

### 4. Action Space

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

### 5. Reward Function

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

### 6. RL Algorithm

Use **Deep Q-Network (DQN)**

- Input: state vector (~21D)
- Output: Q-values for each action

Network structure:

- Linear(21, 128) → ReLU
- Linear(128, 128) → ReLU
- Linear(128, action_dim)

------

### 7. Episode Design

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

### 8. Training Strategy

- Stage 1: Empty environment (learn to move toward goal)
- Stage 2: Add simple obstacles (learn avoidance)
- Stage 3: Evaluate robustness under noise

------

### 9. Noise Experiment

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

### 10. Key Design Choices

- Fix altitude to reduce the problem to 2D
- Use Lidar sectorization to avoid high-dimensional input
- Use DQN to handle continuous state and discrete actions
- Add a safety override using DistanceFront
- Ignore ground collision to prevent reward bias