# config.py

# =========================
# AirSim / Environment
# =========================

INIT_ALTITUDE = -3.0
MAX_STEPS = 600
ACTION_DURATION = 0.2
VELOCITY = 4.0


# =========================
# Training
# =========================

NUM_EPISODES = 300
BATCH_SIZE = 64
GAMMA = 0.99

LEARNING_RATE = 1e-3

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.98

TARGET_UPDATE_FREQ = 10


# =========================
# Map / ROI
# =========================

ROI_CONFIGS = {
    "dense_block": (80, 500, 500, 980),
    "lower_left": (80, 500, 500, 980),
}

TRAINING_ROI_NAME = "lower_left"
TRAINING_ROI = ROI_CONFIGS[TRAINING_ROI_NAME]


# =========================
# Sensor
# =========================

MAX_LIDAR_RANGE = 15.0
MAX_GOAL_DISTANCE = 80.0


# =========================
# Output
# =========================

RUN_BASE_DIR = "runs"
RUN_MODE = "train"
ALGO_NAME = "dqn"
SAVE_TRAJECTORY_EVERY = 5

# =========================
# Plot Colors
# =========================

# OpenCV uses BGR
CV2_COLORS = {
    "goal_reached": (0, 255, 0),   # green
    "collision": (0, 0, 255),      # red
    "timeout": (255, 0, 0),        # blue
    "out_of_roi": (255, 0, 255),   # purple
    "unknown": (0, 0, 0),          # black
    "start": (0, 255, 0),
    "goal": (0, 0, 255),
    "roi": (0, 0, 255),
}

# Matplotlib uses color names
MPL_COLORS = {
    "goal_reached": "green",
    "collision": "red",
    "timeout": "blue",
    "out_of_roi": "purple",
    "unknown": "black",
    "start": "green",
    "goal": "red",
    "roi": "red",
}