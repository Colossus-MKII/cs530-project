import os
import pandas as pd
import matplotlib.pyplot as plt


LOG_PATH = r".\runs\2026-05-02_15-41-36_train_dqn_alt10.0m_lower_left_ep500\training_log.csv"
WINDOW = 20


def save_plot(path):
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {path}")


def plot_curve(df, x_col, y_col, title, ylabel, out_path, window=20):
    ma_col = f"{y_col}_ma"
    df[ma_col] = df[y_col].rolling(window).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[y_col], alpha=0.35, label=y_col)
    plt.plot(df[x_col], df[ma_col], linewidth=2, label=f"{window}-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_plot(out_path)


def main():
    df = pd.read_csv(LOG_PATH)

    # =========================
    # Auto output directory
    # =========================
    experiment_name = os.path.basename(os.path.dirname(LOG_PATH))

    out_dir = os.path.join("analysis_results", experiment_name)

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[INFO] Analysis output directory:")
    print(out_dir)

    print("\n===== Basic Summary =====")
    print("Total episodes:", len(df))
    print(df["result"].value_counts())

    print("\n===== Result Rate =====")
    print(df["result"].value_counts(normalize=True))

    # =========================
    # Reward curve
    # =========================
    plot_curve(
        df,
        "episode",
        "reward",
        "Reward Curve",
        "Reward",
        os.path.join(out_dir, "reward_curve.png"),
        WINDOW
    )

    # =========================
    # Result distribution
    # =========================
    result_counts = df["result"].value_counts()

    plt.figure(figsize=(8, 5))
    result_counts.plot(kind="bar")
    plt.xlabel("Result")
    plt.ylabel("Count")
    plt.title("Episode Result Distribution")
    plt.grid(axis="y")
    save_plot(os.path.join(out_dir, "result_distribution.png"))

    # =========================
    # Rolling result rates
    # =========================
    result_labels = [
        "goal_reached",
        "collision",
        "timeout",
        "out_of_roi",
        "too_low",
        "unknown"
    ]

    for label in result_labels:
        df[f"{label}_rate"] = (df["result"] == label).rolling(WINDOW).mean()

    plt.figure(figsize=(10, 5))
    for label in result_labels:
        plt.plot(df["episode"], df[f"{label}_rate"], label=label)

    plt.xlabel("Episode")
    plt.ylabel(f"Rolling Rate ({WINDOW} episodes)")
    plt.title("Rolling Result Rates")
    plt.legend()
    plt.grid(True)
    save_plot(os.path.join(out_dir, "rolling_result_rates.png"))

    # =========================
    # too_high warning rate
    # =========================
    if "ever_too_high" in df.columns:
        df["ever_too_high_numeric"] = df["ever_too_high"].astype(str).str.lower().isin(["true", "1", "yes"])
        df["too_high_rate"] = df["ever_too_high_numeric"].rolling(WINDOW).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(df["episode"], df["too_high_rate"], linewidth=2, label="ever_too_high rate")
        plt.xlabel("Episode")
        plt.ylabel(f"Rolling Rate ({WINDOW} episodes)")
        plt.title("Rolling Too-High Warning Rate")
        plt.legend()
        plt.grid(True)
        save_plot(os.path.join(out_dir, "rolling_too_high_rate.png"))

        print("\n===== Too High Summary =====")
        print("Episodes ever too high:", int(df["ever_too_high_numeric"].sum()))
        print("Too high rate:", df["ever_too_high_numeric"].mean())

    # =========================
    # Loss curve
    # =========================
    if "avg_loss" in df.columns:
        plot_curve(
            df,
            "episode",
            "avg_loss",
            "DQN Training Loss",
            "Average Loss",
            os.path.join(out_dir, "loss_curve.png"),
            WINDOW
        )

    # =========================
    # Final distance curve
    # =========================
    if "final_distance" in df.columns:
        plot_curve(
            df,
            "episode",
            "final_distance",
            "Final 3D Distance to Goal",
            "Final Distance",
            os.path.join(out_dir, "final_distance_curve.png"),
            WINDOW
        )

    # =========================
    # Final altitude curve
    # =========================
    if "final_altitude" in df.columns:
        plot_curve(
            df,
            "episode",
            "final_altitude",
            "Final Altitude Curve",
            "Final Altitude (m)",
            os.path.join(out_dir, "final_altitude_curve.png"),
            WINDOW
        )

    # =========================
    # Min lidar curve
    # =========================
    if "min_lidar" in df.columns:
        plot_curve(
            df,
            "episode",
            "min_lidar",
            "Minimum LiDAR Distance Curve",
            "Min LiDAR Distance (m)",
            os.path.join(out_dir, "min_lidar_curve.png"),
            WINDOW
        )

    # =========================
    # Steps curve
    # =========================
    if "steps" in df.columns:
        plot_curve(
            df,
            "episode",
            "steps",
            "Episode Length",
            "Steps",
            os.path.join(out_dir, "steps_curve.png"),
            WINDOW
        )

    # =========================
    # Save summary txt
    # =========================
    summary_path = os.path.join(out_dir, "summary.txt")

    with open(summary_path, "w") as f:
        f.write("===== Basic Summary =====\n")
        f.write(f"Total episodes: {len(df)}\n\n")

        f.write("Result counts:\n")
        f.write(str(df["result"].value_counts()))
        f.write("\n\nResult rates:\n")
        f.write(str(df["result"].value_counts(normalize=True)))

        f.write("\n\nReward mean:\n")
        f.write(str(df["reward"].mean()))
        f.write("\nReward max:\n")
        f.write(str(df["reward"].max()))
        f.write("\nReward min:\n")
        f.write(str(df["reward"].min()))

        if "final_distance" in df.columns:
            f.write("\n\nFinal distance mean:\n")
            f.write(str(df["final_distance"].mean()))

        if "final_altitude" in df.columns:
            f.write("\n\nFinal altitude mean:\n")
            f.write(str(df["final_altitude"].mean()))

        if "min_lidar" in df.columns:
            f.write("\n\nMin lidar mean:\n")
            f.write(str(df["min_lidar"].mean()))

        if "ever_too_high" in df.columns:
            ever_too_high = df["ever_too_high"].astype(str).str.lower().isin(["true", "1", "yes"])
            f.write("\n\nEpisodes ever too high:\n")
            f.write(str(int(ever_too_high.sum())))
            f.write("\nToo high rate:\n")
            f.write(str(ever_too_high.mean()))

    print(f"[SAVE] {summary_path}")
    print("\nAnalysis finished.")


if __name__ == "__main__":
    main()