"""config_legacy.py — 批量处理流水线配置（供 main.py 使用）

功能：定义视频输入路径、输出路径、模型路径及各项处理参数
"""
import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

# 路径配置
VIDEO_PATH = os.path.join(_PROJECT_DIR, "videos")
OUTPUT_DIR = os.path.join(_PROJECT_DIR, "data", "rallies_new")
MODEL_PATH = os.path.join(_PROJECT_DIR, "models", "yolo", "yolo11x-pose.pt")

# 视频处理与队列参数
SCOUT_SKIP_FRAMES = 5
SCOUT_SCALE = 0.5
MIN_RALLY_DURATION = 8.0  # 最小片段时长(秒)

# 姿态追踪参数
YOLO_IMGSZ = 1024
CONF_FAR = 0.15
CONF_NEAR = 0.3
POSE_MAX_GAP = 5          # 丢帧补偿上限(帧)
POSE_ALPHA = 0.6          # EMA 平滑系数

# 控制文件与存档
CONTROL_FILE = os.path.join(_PROJECT_DIR, "logs", "control.txt")
CHECKPOINT_FILE = os.path.join(_PROJECT_DIR, "logs", "checkpoint.json")