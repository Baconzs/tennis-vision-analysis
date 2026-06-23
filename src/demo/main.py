"""main.py — Demo 入口"""
import sys
import os
import argparse

# torch 必须在 PyQt5 之前 import，否则 Windows 下 CUDA DLL 加载顺序冲突
import torch  # noqa: F401

# 确保 src/demo 在路径中
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

# 修复 Windows 下 PyQt5 找不到 platform plugin 的问题
import PyQt5
_qt_plugins = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
if os.path.isdir(_qt_plugins):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt_plugins

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from app import MainWindow


def main():
    parser = argparse.ArgumentParser(description="网球动作识别 Demo")
    parser.add_argument("--rally",   default="", help="Rally 目录路径")
    parser.add_argument("--config",  default="", help="配置 YAML 路径")
    parser.add_argument("--weights", default="", help="模型权重 .pth 路径")
    parser.add_argument("--person",  default="", help="person 检测 YOLO .pt 路径")
    parser.add_argument("--pose",    default="", help="pose 估计 YOLO .pt 路径")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))

    win = MainWindow(
        rally_dir=args.rally   or None,
        config_path=args.config or None,
        weights_path=args.weights or None,
        person_model=args.person or None,
        pose_model=args.pose   or None,
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
