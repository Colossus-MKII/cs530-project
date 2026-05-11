# Reward Function Design

The reward function was designed to balance four major objectives:

1. Goal-directed navigation  
2. Obstacle avoidance  
3. Energy-efficient flight  
4. Safe altitude control  

The total reward is defined as:

$$
R(s,a)=R_{progress}+R_{step}+R_{obstacle}+R_{goal}+R_{altitude}+R_{action}+R_{collision}
$$

Although both environments share the same overall reward structure, the soft and hard environments apply different safety constraints and termination strategies.

---

# Soft Environment Reward Design

The soft environment contains relatively sparse obstacles and primarily focuses on encouraging efficient navigation behavior.

## Progress Reward

$$
R_{progress}=5(d_{t-1}-d_t)
$$

where:

- $d_t$ is the current distance to the goal
- $d_{t-1}$ is the previous distance

This term rewards the agent for moving closer to the target while penalizing backward movement.

---

## Step Penalty

$$
R_{step}=-0.05
$$

A small negative reward is applied at every timestep to encourage shorter and more efficient trajectories.

---

## Obstacle Penalty

$$
R_{obstacle}=
\begin{cases}
-15 & d_{lidar}<0.8\\
-3 & 0.8\le d_{lidar}<1.5\\
0 & \text{otherwise}
\end{cases}
$$

where $d_{lidar}$ represents the distance to the nearest obstacle.

Obstacle penalties remain relatively moderate in the soft environment because obstacle density is low.

---

## Goal Reward

$$
R_{goal}=
\begin{cases}
100 & d_t<1.0\\
0 & \text{otherwise}
\end{cases}
$$

A large terminal reward encourages successful navigation completion.

---

## Altitude Cost

$$
R_{altitude}=-0.02h
$$

where $h$ represents the drone altitude.

This term discourages unnecessary climbing and promotes energy-efficient flight.

---

## Collision Penalty

$$
R_{collision}=
\begin{cases}
-100 & \text{if collision}\\
0 & \text{otherwise}
\end{cases}
$$

Collisions are heavily penalized in both environments.

---

## Soft Environment Constraint Strategy

In the soft environment, unsafe altitude behavior is treated as a soft penalty rather than a terminal failure.

### High Altitude Penalty

$$
R_{high}=
\begin{cases}
-20 & h>20\\
0 & \text{otherwise}
\end{cases}
$$

### Low Altitude Penalty

$$
R_{low}=
\begin{cases}
-50 & h<2\\
0 & \text{otherwise}
\end{cases}
$$

These penalties discourage unsafe altitude behavior while allowing the episode to continue.

The soft environment therefore prioritizes efficient navigation and stable exploration instead of strict safety enforcement.

---

# Hard Environment Reward Design

The hard environment contains dense obstacles and narrow passages. Therefore, the reward design places greater emphasis on safety and adaptive obstacle avoidance behavior.

---

## Enhanced Obstacle Avoidance

The same obstacle penalty structure is retained:

$$
R_{obstacle}=
\begin{cases}
-15 & d_{lidar}<0.8\\
-3 & 0.8\le d_{lidar}<1.5\\
0 & \text{otherwise}
\end{cases}
$$

However, obstacle interactions occur significantly more frequently due to the increased environmental complexity.

---

## Context-Aware Vertical Action Reward

To encourage adaptive obstacle avoidance behavior, the hard environment introduces action-dependent rewards.

### Upward Reward

$$
R_{up}=
\begin{cases}
+0.8 & d_{lidar}<2.0 \text{ and } a=up\\
0 & \text{otherwise}
\end{cases}
$$

### Downward Penalty

$$
R_{down}=
\begin{cases}
-2.0 & d_{lidar}<2.0 \text{ and } a=down\\
0 & \text{otherwise}
\end{cases}
$$

These rewards encourage the drone to climb when nearby obstacles are detected while discouraging unsafe downward motion.

---

## Hard Environment Terminal Constraints

Unlike the soft environment, the hard environment treats invalid flight behaviors as terminal failures.

If the drone:

- leaves the valid flight boundary
- exceeds the maximum allowed altitude
- collides with an obstacle

the episode immediately terminates and training restarts.

The terminal condition is defined as:

$$
done=
\begin{cases}
True & \text{if out of boundary}\\
True & \text{if } h>h_{max}\\
True & \text{if collision}\\
False & \text{otherwise}
\end{cases}
$$

This design prevents the agent from exploiting the reward function through unrealistic escape strategies, such as flying excessively high to avoid the obstacle field entirely.

Without terminal constraints, the agent may prioritize reward exploitation instead of learning meaningful obstacle avoidance behavior.

Therefore, the hard environment emphasizes survival and safety constraints rather than purely shortest-path navigation.

---

# Final Reward Philosophy

The reward system encourages the agent to:

- Move toward the target efficiently
- Avoid nearby obstacles
- Perform vertical maneuvers only when necessary
- Maintain energy-efficient flight
- Avoid collisions at all costs

The soft environment prioritizes navigation efficiency and smoother exploration, while the hard environment prioritizes safety and adaptive maneuvering under dense obstacle conditions.

This difference significantly changes the exploration difficulty and learning stability of the reinforcement learning agent.