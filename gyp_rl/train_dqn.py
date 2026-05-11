import os
import csv
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import config
from airsim_env import AirSimDroneEnv


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


def select_action(policy_net, state, epsilon, action_dim, device):
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = policy_net(state_tensor)

    return int(torch.argmax(q_values).item())


def train_step(policy_net, target_net, replay_buffer, optimizer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    q_values = policy_net(states)
    q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(1)[0]
        target_q = rewards + gamma * max_next_q * (1 - dones)

    loss = nn.MSELoss()(q_action, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def main():
    num_episodes = 800
    batch_size = 64
    gamma = 0.99

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    target_update_freq = 5

    success_count = 0
    collision_count = 0
    timeout_count = 0
    too_low_count = 0
    out_of_roi_count = 0
    too_high_count = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = AirSimDroneEnv(
        init_altitude=-10.0,
        max_steps=600,
        action_duration=0.2,
        velocity=4.0,
    )

    print("Training ROI:", env.training_roi)

    env.sampler.init_run_dir(
        mode="train",
        algo="dqn",
        extra_info=f"alt{abs(env.init_altitude):.1f}m_{config.TRAINING_ROI_NAME}_ep{num_episodes}"
    )
    env.sampler.save_training_roi_plot(env.training_roi)

    log_path = os.path.join(env.sampler.run_dir, "training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "start_x", "start_y", "start_z",
            "goal_x", "goal_y", "goal_z",
            "reward",
            "steps",
            "epsilon",
            "avg_loss",
            "final_distance",
            "final_altitude",
            "min_lidar",
            "ever_too_high",
            "ever_out_of_roi",
            "result"
        ])

    state_dim = env.state_dim
    action_dim = env.action_dim

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=50000)

    try:
        for episode in range(num_episodes):
            print(f"\n===== Episode {episode + 1} / {num_episodes} =====")

            episode_too_high = False
            episode_out_of_roi = False

            state = env.reset(mode="train")

            path = []
            path.append(env.get_position_xyz())

            total_reward = 0.0
            total_loss = 0.0
            loss_count = 0
            info = {}

            for step in range(env.max_steps):
                action = select_action(
                    policy_net,
                    state,
                    epsilon,
                    action_dim,
                    device
                )

                next_state, reward, done, info = env.step(action)

                if info.get("too_high"):
                    episode_too_high = True

                if info.get("out_of_roi"):
                    episode_out_of_roi = True

                # current_pos = env.get_position_xy()
                current_pos = env.get_position_xyz()
                path.append(current_pos)

                replay_buffer.push(
                    state,
                    action,
                    reward,
                    next_state,
                    float(done)
                )

                loss = train_step(
                    policy_net,
                    target_net,
                    replay_buffer,
                    optimizer,
                    batch_size,
                    gamma,
                    device
                )

                if loss is not None:
                    total_loss += loss
                    loss_count += 1

                state = next_state
                total_reward += reward

                if done:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if (episode + 1) % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
            final_distance = env._distance_to_goal()

            if info.get("goal_reached"):
                result = "goal_reached"
                success_count += 1
            elif info.get("collision"):
                result = "collision"
                collision_count += 1
            elif info.get("timeout"):
                result = "timeout"
                timeout_count += 1
            elif info.get("too_low"):
                result = "too_low"
                too_low_count += 1
            elif info.get("too_high"):
                result = "too_high"
                too_high_count += 1
            elif info.get("out_of_roi"):
                result = "out_of_roi"
                out_of_roi_count += 1
            else:
                result = "unknown"

            if (episode + 1) % 5 == 0:
                env.sampler.save_altitude_plot(
                    path=path,
                    episode=episode + 1,
                    result=result,
                    min_flight_altitude=env.min_flight_altitude,
                    max_flight_altitude=env.max_flight_altitude
                )

                env.sampler.save_trajectory_plot(
                    start=env.start,
                    goal=env.goal,
                    path=path,
                    episode=episode + 1,
                    roi=env.training_roi,
                    result=result
                )

                env.sampler.save_trajectory_3d_plot(
                    start=env.start,
                    goal=env.goal,
                    path=path,
                    episode=episode + 1,
                    result=result,
                    min_flight_altitude=env.min_flight_altitude,
                    max_flight_altitude=env.max_flight_altitude
                )

            finished_episodes = episode + 1

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode + 1,

                    env.start[0],
                    env.start[1],
                    env.start[2],

                    env.goal[0],
                    env.goal[1],
                    env.goal[2],

                    total_reward,
                    step + 1,
                    epsilon,
                    avg_loss,
                    final_distance,
                    info.get("altitude", None),
                    info.get("min_lidar", None),
                    episode_too_high,
                    episode_out_of_roi,
                    result
                ])

            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Steps: {step + 1} | "
                f"Loss: {avg_loss:.4f} | "
                f"Result: {result}"
            )

        model_name = f"model_ep{num_episodes}.pth"
        save_path = os.path.join(env.sampler.run_dir, model_name)
        torch.save(policy_net.state_dict(), save_path)
        print(f"\n[MODEL SAVED] {save_path}")

    finally:
        env.close()

if __name__ == "__main__":
    main()