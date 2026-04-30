import time
from airsim_env import AirSimDroneEnv


def run_action_sequence(env, name, actions, start, goal):
    print(f"\n========== Test: {name} ==========")

    state = env.reset(start=start, goal=goal, mode="train")

    print("Start:", start)
    print("Goal:", goal)
    print("Initial state shape:", state.shape)
    print("Initial distance:", env._distance_to_goal())
    print("Initial altitude_norm:", state[2])
    print("Initial lidar:", state[3:21])

    total_reward = 0.0

    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        print(f"\nStep {i + 1}")
        print("Action:", action)
        print("Distance:", env._distance_to_goal())
        print("Altitude norm:", next_state[2])
        print("Reward:", reward)
        print("Total reward:", total_reward)
        print("Done:", done)

        terms = info.get("reward_terms", {})

        print("Reward breakdown:")
        for k, v in terms.items():
            if abs(v) > 1e-6:
                print(f"  {k}: {v:.4f}")

        print("Min lidar:", info.get("min_lidar"))
        print("Altitude:", info.get("altitude"))
        print("Progress:", info.get("progress"))

        if done:
            break

        time.sleep(0.2)

    print(f"\nTest finished: {name}")
    print("Final total reward:", total_reward)


def main():
    env = AirSimDroneEnv(
        init_altitude=-3.0,
        max_steps=50,
        action_duration=0.2,
        velocity=4.0,
    )

    start = (0.0, 0.0)
    goal = (20.0, 0.0)

    try:
        # action mapping:
        # 0 forward
        # 1 forward-left
        # 2 forward-right
        # 3 left
        # 4 right
        # 5 back
        # 6 hover
        # 7 up
        # 8 down

        run_action_sequence(
            env,
            name="Move toward goal",
            actions=[0, 0, 0, 0, 0],
            start=start,
            goal=goal,
        )

        run_action_sequence(
            env,
            name="Move away from goal",
            actions=[5, 5, 5, 5, 5],
            start=start,
            goal=goal,
        )

        run_action_sequence(
            env,
            name="Hover",
            actions=[6, 6, 6, 6, 6],
            start=start,
            goal=goal,
        )

        run_action_sequence(
            env,
            name="Side movement",
            actions=[3, 3, 4, 4],
            start=start,
            goal=goal,
        )

        run_action_sequence(
            env,
            name="Move up",
            actions=[7, 7, 7, 7, 7],
            start=start,
            goal=goal,
        )

        run_action_sequence(
            env,
            name="Move down",
            actions=[8, 8, 8, 8, 8],
            start=start,
            goal=goal,
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()