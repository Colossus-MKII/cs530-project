# RL Training Plan

# 1. Project Objective

The goal of this project is to train a reinforcement learning agent for autonomous drone navigation in complex environments using AirSim.

The agent must:

- Navigate toward a target location
- Avoid static obstacles
- Maintain safe flight behavior
- Learn adaptive maneuvering strategies
- Generalize across different environments

The long-term objective is to develop a robust navigation framework that can operate under varying obstacle densities, altitude constraints, and sensor conditions.

---

# 2. Training Environment Design

The training environments are divided into two difficulty levels:

## Soft Environment

The soft environment contains:

- Sparse obstacles
- Wider free-space regions
- Lower navigation complexity

Purpose:

- Learn basic goal-directed navigation
- Learn simple obstacle avoidance
- Stabilize early-stage exploration

The soft environment focuses on improving training stability and sample efficiency.

---

## Hard Environment

The hard environment contains:

- Dense obstacles
- Narrow passages
- Higher collision probability
- Stronger safety constraints

Purpose:

- Learn adaptive obstacle avoidance
- Learn vertical maneuvering behavior
- Improve robustness under constrained environments

The hard environment significantly increases exploration difficulty and learning complexity.

---

# 3. Sensor Configuration

## Primary Sensor: Lidar

The main RL observation is generated using Lidar sectorization.

### Processing Pipeline

- 180° forward-facing field of view
- Divided into 18 angular sectors
- Minimum obstacle distance extracted from each sector
- Distance clipped to 15 meters
- Normalized to [0,1]

Output:

```python
lidar_state = [d1, d2, ..., d18]
```

This representation reduces state dimensionality while preserving obstacle structure information.

---

## Secondary Sensor: DistanceFront

Used for:

- Emergency safety checks
- Additional obstacle proximity estimation

---

## DepthPlanar Camera

Currently used only for:

- Visualization
- Debugging
- Trajectory inspection

Not yet included in RL state input.

---

# 4. State Representation

The current state vector is:

```python
state = [
    goal_angle,
    distance_to_goal,
    lidar_sector_1,
    ...,
    lidar_sector_18,
    distance_front
]
```

State dimension:

```text
≈ 21 dimensions
```

Normalization:

- goal_angle / π
- distance_to_goal / max_range
- lidar / 15
- distance_front / 15

---

# 5. Action Space

The current implementation uses discrete actions:

- forward
- forward-left
- forward-right
- left
- right
- back
- hover

Each action is executed for a fixed duration.

Current action duration:

```text
0.5 seconds
```

---

# 6. Reward Function Design

The reward function balances:

- Goal progress
- Obstacle avoidance
- Energy-efficient flight
- Safety constraints

General form:

$$
R(s,a)=R_{progress}+R_{goal}+R_{collision}+R_{danger}+R_{step}
$$

---

## Soft Environment Reward Philosophy

The soft environment mainly encourages:

- Efficient navigation
- Stable exploration
- Basic obstacle avoidance

Unsafe behavior is penalized but does not terminate the episode immediately.

---

## Hard Environment Reward Philosophy

The hard environment emphasizes:

- Safety-critical behavior
- Adaptive vertical maneuvering
- Strong obstacle avoidance

Boundary violations and excessive altitude are treated as terminal failures.

This prevents reward exploitation strategies such as escaping vertically instead of solving the navigation task properly.

---

# 7. RL Algorithm

## Current Algorithm: DQN

The current implementation uses Deep Q-Network (DQN).

Input:

```text
~21-dimensional state vector
```

Output:

```text
Q-values for discrete actions
```

Current network structure:

```text
Linear(21,128)
ReLU
Linear(128,128)
ReLU
Linear(128,action_dim)
```

---

# 8. Episode Design

## Reset Procedure

At the beginning of each episode:

1. Reset AirSim environment
2. Take off
3. Move to predefined altitude
4. Initialize start and goal positions
5. Read initial sensor state

---

## Termination Conditions

Episodes terminate when:

- Goal reached
- Collision occurs
- Boundary violation occurs
- Maximum altitude exceeded
- Maximum step count reached

---

# 9. Multi-Stage Training Strategy

The project follows a curriculum-style training strategy.

---

## Stage 1: Soft Environment Training

Objective:

- Learn stable navigation behavior
- Learn basic obstacle avoidance
- Improve exploration stability

---

## Stage 2: Hard Environment Training

Objective:

- Learn dense obstacle navigation
- Improve survival behavior
- Learn adaptive maneuvering

---

## Stage 3: Multi-Altitude Training (Future Work)

Future experiments will introduce:

- Different flight altitudes
- Dynamic altitude switching
- Variable obstacle height distributions

Purpose:

- Improve generalization capability
- Reduce overfitting to fixed-altitude environments

---

## Stage 4: CNN-Based Visual Navigation (Future Work)

Future work will integrate image-based observations.

Possible inputs:

- DepthPlanar
- Segmentation camera
- RGB images

Potential architecture:

```text
CNN + DQN
```

Purpose:

- Learn spatial obstacle structure
- Improve visual navigation capability
- Replace hand-crafted lidar sectorization

---

## Stage 5: Cross-Environment Generalization Testing (Future Work)

Future experiments will evaluate:

- Training in one environment
- Testing in unseen environments

Metrics:

- Success rate
- Collision rate
- Path efficiency
- Robustness

Purpose:

- Evaluate policy generalization
- Measure transfer capability

---

# 10. Noise Robustness Experiment

Gaussian noise will be added to Lidar observations:

$$
d_{noisy}=d+\mathcal{N}(0,\sigma)
$$

Test settings:

- σ = 0
- σ = 0.1
- σ = 0.3
- σ = 0.5

Evaluation metrics:

- Success rate
- Collision rate
- Average reward
- Path length

---

# 11. Long-Term Research Direction

Future improvements may include:

- Double DQN
- PPO
- SAC
- Hierarchical RL
- Multi-agent coordination
- Sim-to-real transfer
- Sensor fusion
- Transformer-based visual policies

The overall objective is to build a scalable and robust autonomous navigation framework for complex drone environments.