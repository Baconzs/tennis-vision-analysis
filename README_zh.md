# 网球比赛视频视觉分析系统

> 基于计算机视觉的网球比赛自动分析系统 —— 从原始比赛视频中自动检测球场、追踪球员姿态，并识别球员动作（待机 / 正手 / 反手 / 发球 / 移动）。

[English](./README.md) | 简体中文

---

## 简介

本项目是一套面向网球单打比赛视频的端到端视觉分析流水线，整体分为三个串联阶段：

```
原始比赛视频
   │
   ├─[1] 球场检测       YOLO 关键点 → 14 个球场点 → 计算单应矩阵（俯视坐标）
   │
   ├─[2] 球员姿态追踪    YOLO-pose 检测近端/远端球员 → 17 点骨架 → EMA 平滑、丢帧补偿
   │
   └─[3] 动作识别       自研 MSTFormer → 5 类动作分类 + 关键帧检测（双头输出）
```

其中第 3 阶段的 **MSTFormer**（Multi-Stream Transformer）是本项目的核心：它融合「球员姿态序列 + 球场几何位置 + 多路视觉裁剪图」三种信息，用 Transformer 同时完成**动作分类**与**关键帧检测**两个任务。

## 功能模块

| 模块 | 目录 | 说明 |
| --- | --- | --- |
| **球场检测** | `src/court_detector.py`、`src/pipeline/` | YOLO 14 关键点模型，检测球场角点并计算单应矩阵 |
| **球员姿态追踪** | `src/pose_tracker.py` | YOLO-pose 追踪近/远端球员，含 EMA 平滑与丢帧补偿 |
| **动作识别（核心）** | `src/model/mst/` | MSTFormer：5 类动作 + 关键帧双头，支持视觉 token 三路合并、姿态/裁剪图消融开关 |
| **人员分类** | `src/model/yolo/`、`src/training/` | 区分近端 / 远端球员的 YOLO 分类模型 |
| **批量处理流水线** | `src/main.py`、`src/pipeline/offline_tennis_tracker.py` | 遍历视频，跑完整追踪流水线，支持断点续跑 |
| **可视化 Demo** | `src/demo/` | PyQt5 桌面应用：视频播放 + 三行时间轴（标注/预测/帧）+ 实时推理可视化 |
| **标注与数据工具** | `src/utils/` | 动作时序标注（Flask Web）、球场关键点标注（GUI）、人员框标注、数据集划分等 |

> 各模块都带独立 README（[src](./src/README.md)、[MSTFormer](./src/model/mst/README.md)、[pipeline](./src/pipeline/README.md)、[demo](./src/demo/README.md)、[utils](./src/utils/README.md)、[configs](./configs/README.md)）；逐文件职责见 [`docs/architecture_zh.md`](./docs/architecture_zh.md)；代码约定见 [`docs/代码规范.md`](./docs/代码规范.md)。

## 目录结构

```
tennis-vision-analysis/
├── src/                    源代码
│   ├── main.py             批量视频处理主入口
│   ├── court_detector.py   球场检测器
│   ├── pose_tracker.py     姿态追踪器
│   ├── train_court_pipeline.py  球场模型训练入口
│   ├── pipeline/           离线追踪流水线、数据集准备、标注精修工具
│   ├── model/
│   │   ├── mst/            MSTFormer 模型、训练与评估
│   │   └── yolo/           人员分类模型
│   ├── demo/               PyQt5 可视化 Demo
│   ├── utils/              标注与数据处理脚本
│   └── training/           人员检测/分类训练脚本
├── configs/                YAML 配置（球场、人员、MSTFormer 主配置/消融/超参/组件对比）
├── docs/
│   ├── architecture_zh.md  各文件详细说明与模块依赖
│   └── figures/            实验结果图（训练曲线、混淆矩阵、消融对比等）
├── requirements.txt
├── LICENSE
└── README.md / README_zh.md
```

## 环境安装

```bash
# 建议 Python 3.10+（开发于 3.11 / 3.12）
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

> **PyTorch / CUDA**：`torch`、`torchvision` 请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 指引安装与本机 CUDA 匹配的版本，再安装其余依赖。

## 使用说明

> ⚠️ 本仓库**仅包含源代码、配置与文档**。原始比赛视频、标注数据集、模型权重因体积过大（合计上百 GB）未纳入版本控制，需自行准备并放入对应目录（`videos/`、`data/`、`models/`）。各路径约定见 `src/config_legacy.py` 与 `configs/`。

```bash
# 1) 批量处理视频，输出带标注的追踪结果
python src/main.py

# 2) 训练球场关键点检测模型
python src/train_court_pipeline.py

# 3) 训练 MSTFormer 动作识别模型（指定配置）
python src/model/mst/train.py --config configs/main.yaml

# 4) 启动动作时序标注工具（浏览器访问 http://localhost:5000）
python src/utils/action_annotator.py

# 5) 启动可视化 Demo
python src/demo/main.py --rally <回合目录>
```

## 数据格式

**动作标注 `annotations.json`** —— 每个时间段一条记录：

```json
[
  {"start_time": 0.0,   "end_time": 4.837, "action_name": "待机", "action_id": 0},
  {"start_time": 4.837, "end_time": 12.78, "action_name": "发球", "action_id": 3}
]
```

动作类别：`待机(0)`、`正手(1)`、`反手(2)`、`发球(3)`、`移动(4)`。

**姿态数据 `pose_data.json`** —— 每帧一条记录：

```json
{
  "frame": 0,
  "court": [[x, y, conf], ...],
  "near_player": {"bbox": [x1,y1,x2,y2], "keypoints": [[x,y,conf], ...]},
  "far_player":  {"bbox": [x1,y1,x2,y2], "keypoints": [[x,y,conf], ...]}
}
```

- `court`：14 个球场关键点，置信度 < 0.3 时在特征向量中置零
- `near_player` / `far_player`：17 点 COCO 骨架

## 实验结果

| 训练曲线 | 主混淆矩阵 |
| --- | --- |
| ![训练曲线](./docs/figures/fig1_main_training_curve.png) | ![混淆矩阵](./docs/figures/fig7_confusion_matrix_main.png) |

更多结果（消融实验、超参对比、组件对比、关键帧检测曲线等）见 [`docs/figures/`](./docs/figures/)。

## 技术栈

- **深度学习**：PyTorch、Ultralytics YOLO11（检测 / 姿态 / 关键点）
- **自研模型**：MSTFormer（多流 Transformer，姿态 + 球场几何 + 视觉裁剪图）
- **CV / 数值**：OpenCV、NumPy、SciPy、scikit-learn
- **应用 / 标注**：PyQt5（桌面 Demo）、Flask（Web 标注）

## 许可证

本项目采用 [MIT License](./LICENSE)，版权所有 © 2026 Da_233。

---

> 本项目源于毕业设计。开源版本仅含代码与文档，论文正文及受版权保护的参考文献不在其中。
