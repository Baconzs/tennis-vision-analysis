# pipeline/ — 离线精追踪与球场标注工具

本目录两类内容：① 单回合的**离线精追踪**（比 `main.py` 的批量巡场更精细，含球场单应矩阵与雷达图）；② 球场 14 关键点的**标注采样与精修工具**。

## 离线追踪

| 文件 | 作用 | 运行 |
| --- | --- | --- |
| `offline_tennis_tracker.py` | **核心**。两遍处理：Pass 1 用球场关键点模型逐帧算加权单应矩阵、botsort 追踪球员并投影到球场坐标；Pass 2 渲染球场线、球员框与俯视雷达图，输出标注视频 | `python src/pipeline/offline_tennis_tracker.py` |
| `generate_trajectory.py` | 从追踪结果提取球员坐标序列，生成动作识别用的时序轨迹 | `python src/pipeline/generate_trajectory.py` |
| `debug_vision.py` | 可视化调试：把球场检测/球员追踪叠加到视频帧，验证流水线（顶部常量改输入视频/模型路径） | `python src/pipeline/debug_vision.py` |
| `test_weighted_inference.py` | 用 OpenCV 追踪器测试目标追踪效果 | — |

## 球场关键点标注工具

整套球场模型的数据生产流程：

```
比赛视频 ─▶ smart_extract_14pts.py（智能采样 + 模型预标注14点）
              │
              ▼
        corner_driven_refine_tool.py（拖4角点自动算其余10点，手工精修）
              │
              ▼
        prepare_weighted_dataset.py（合并新旧标注，按比例划分 train/val）
              │
              ▼
        ../train_court_pipeline.py（训练新球场模型）
```

| 文件 | 作用 |
| --- | --- |
| `smart_extract_14pts.py` | 从视频智能采样帧，用现有球场模型预标注 14 个关键点，生成训练候选 |
| `corner_driven_refine_tool.py` | 交互式 GUI：拖拽 4 个角点自动推算其余 10 点，支持负样本标注 |
| `prepare_weighted_dataset.py` | 合并新标注与旧数据集，划分 train/val，供球场模型微调 |

> 球场 14 点的物理坐标（米）定义见 `offline_tennis_tracker.py` 与 `../train_court_pipeline.py` 顶部的 `COURT_14_PTS_PHYSICAL`。
