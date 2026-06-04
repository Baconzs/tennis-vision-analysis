# 网球比赛分析项目 — 文件清单

> 毕业设计：基于计算机视觉的网球比赛自动分析系统
> 更新：2026-04-24（session18：配置体系重组、位置编码开关、inference.py 修复）

---

## 目录结构总览

```
目录结构总览
项目标注与测试/
├── src/                    源代码
│   ├── pipeline/           核心追踪流水线
│   ├── model/              动作识别模型
│   │   └── mst/            MSTFormer 模型代码（独立目录）
│   ├── demo/               PyQt5 Demo 应用（视频播放 + 推理可视化）
│   ├── utils/              工具脚本（标注、数据处理）
│   ├── main.py             批量处理主入口
│   ├── train_court_pipeline.py  球场模型训练入口
│   ├── court_detector.py   球场检测器模块
│   ├── pose_tracker.py     姿态追踪器模块
│   └── config_legacy.py    批量处理配置
├── models/                 模型权重
│   ├── yolo/               YOLO 系列通用模型
│   ├── court/              球场关键点检测模型
│   ├── action/             动作识别模型
│   └── person/             人员分类模型
├── data/                   数据集
│   ├── rallies_annotated/  标注回合（含 annotations.json）
│   ├── rallies_new/        新采集比赛回合数据
│   ├── court_finetune/     球场微调数据集
│   └── person_sorter/      人员分类数据集
├── videos/                 原始比赛视频
├── configs/                YAML 训练配置文件
├── runs/                   训练运行记录
├── results/                分析结果与演示视频
├── logs/                   流水线运行日志
└── _archive/               归档区（旧代码/旧数据）
    ├── legacy_src/         旧版源代码
    └── trainData/          旧版训练数据
```

---

## 一、源代码 (`src/`)

### 主入口

| 文件 | 用途 |
| --- | --- |
| `src/main.py` | 批量视频处理主入口。遍历 `videos/` 目录，对每场比赛视频运行完整追踪流水线，支持断点续跑（依赖 `config_legacy.py`、`court_detector.py`、`pose_tracker.py`） |
| `src/train_court_pipeline.py` | 球场关键点模型训练入口。准备数据集 YAML → 启动 YOLO 微调训练 → 导出 Bad Cases 用于迭代优化 |
| `src/config_legacy.py` | `main.py` 的配置文件。定义视频输入路径、输出路径、模型路径及处理参数（帧跳过数、置信度阈值等） |
| `src/court_detector.py` | 球场检测器类（`CourtDetector`）。用霍夫直线快速检测球场并框出远/近端 ROI（巡场阶段用，非关键点模型），供 `main.py` 调用 |
| `src/pose_tracker.py` | 姿态追踪器类（`PoseTracker`）。封装 YOLO 姿态估计，含 EMA 平滑和丢帧补偿，供 `main.py` 调用 |

### 核心追踪流水线 (`src/pipeline/`)

| 文件 | 用途 |
| --- | --- |
| `offline_tennis_tracker.py` | **核心模块**。离线网球追踪：读取回合视频，用球场关键点模型计算单应矩阵，结合姿态模型追踪球员，输出带标注的视频 |
| `generate_trajectory.py` | 轨迹生成模块。从追踪结果中提取球员坐标序列，生成用于动作识别的时序特征 |
| `debug_vision.py` | 可视化调试工具。将球场检测、球员追踪结果叠加到视频帧上，用于验证流水线输出 |
| `smart_extract_14pts.py` | 智能采样标注工具。从比赛视频中智能采样帧，用球场模型预标注14个关键点，生成训练数据 |
| `corner_driven_refine_tool.py` | 球场标注精修工具（交互式 GUI）。拖拽4个角点，自动计算其余10个关键点，支持负样本标注 |
| `prepare_weighted_dataset.py` | 数据集合并工具。将新标注数据与旧数据集合并，按比例划分 train/val，用于球场模型微调 |
| `test_weighted_inference.py` | 推理测试脚本。用 OpenCV 追踪器测试视频中的目标追踪效果，验证推理流程 |

### 动作识别模型 (`src/model/mst/`)

| 文件 | 用途 |
| --- | --- |
| `model_main.py` | MSTFormer 模型定义。双头输出：5类动作分类 + 关键帧二分类。支持 `merge_visual_tokens`（三路合并 resampler）、`use_player_crops`（是否使用 p1/p2 裁剪图）、`use_pose`（姿态 token 是否置零）开关。physics_extractor 输入 125 维 |
| `modules/action_head.py` | 分类头模块。`ActionClassificationHead`（5类动作）+ `KeyframeDetectionHead`（关键帧二分类），结构相同均为 LN→Linear→GELU→Dropout→Linear |
| `modules/pos_encoding.py` | 位置编码模块。为 Transformer 提供正弦位置编码 |
| `modules/backbone_factory.py` | 视觉骨干工厂。根据 `visual_backbone` 配置构建 YOLO11 / ViT 特征提取器 |
| `modules/token_resampler.py` | Token 重采样模块。将任意数量 token 压缩到固定数量（cross-attention resampler） |
| `tokenizer_pseudo.py` | 伪 tokenizer 说明文档。描述将姿态序列转为 token 的方式 |
| `dataset.py` | 动作识别数据集加载器。读取 `annotations.json`，构建滑动窗口序列，返回 `(pose, packed_frames, action_labels, keyframe_labels)`。pose 向量 125 维（含球场14点坐标、人物相对球场中心位置）。支持 `image_augment` 图像增强：颜色抖动/高斯噪声/模糊/随机擦除/半透明覆盖 |
| `config.py` | 模型训练超参数配置（旧版，已被 yaml 替代） |
| `train.py` | MSTFormer 训练脚本。联合训练动作分类 + 关键帧检测，每轮输出动作 Acc 和关键帧 Precision/Recall。每次训练在 `models/action/<config>/<timestamp>/` 下保存 `best.pth`、`final.pth`、`train_log.csv`（每轮指标）、`config.yaml`（配置快照） |
| `test_matrix.py` | 混淆矩阵评估脚本 |
| `test_dataset.py` | 数据集加载验证脚本 |
| `tests/eval_optimal.py` | 模型评估脚本，支持 `--config`/`--weights` 参数，生成混淆矩阵图片+CSV+分类报告 |

### Demo 应用 (`src/demo/`)

| 文件 | 用途 |
| --- | --- |
| `main.py` | Demo 入口。解析命令行参数（`--rally`、`--config`、`--weights`、`--person`、`--pose`），修复 Windows CUDA DLL 加载顺序问题 |
| `app.py` | 主窗口（PyQt5）。视频播放、三行时间轴、文件选择、YOLO 模型路径输入、推理触发、动作图例 |
| `player.py` | 视频播放器。QTimer + OpenCV 逐帧读取，处理中文路径（短路径转换） |
| `timeline.py` | 时间轴面板。三行：GT 标注条 / 预测条 / 帧格子条，游标跟随播放位置 |
| `inference.py` | 推理线程（QThread）。支持两种模式：① 填入 person/pose YOLO 路径时实时检测，绘制 bbox + 骨架叠加到帧上；② 不填时回退读 `pose_data.json` + 预提取裁剪图。两种模式均将全序列一次性送入 MSTFormer |
| `seq_len_sweep.py` | 序列长度扫描脚本。遍历不同 seq_len，输出准确率 CSV |

### 工具脚本 (`src/utils/`)

| 文件 | 用途 |
| --- | --- |
| `action_annotator.py` | **动作时序标注工具**（Flask Web 应用）。从 `data/rallies_new/` 按视频来源轮换提取片段到 `data/rallies_annotating/`，在浏览器中标注5类动作时间段，保存为 `annotations.json`。支持删除片段（自动补充新片段）、进度持久化（`_progress.json` 记录 deleted，重启不重复抽取）。用法：`python action_annotator.py`，访问 `http://localhost:5000` |
| `label_tool.py` | 球员边界框标注工具（OpenCV GUI）。在图片上拖拽绘制 bounding box，标注近端/远端球员，保存为 YOLO txt 格式 |
| `data-batch-extractor.py` | 批量回合数据提取流水线。遍历 `data/rallies_new/`，对每个回合视频运行球场检测 + 姿态追踪，输出 `tracking_data.json`，支持进度记录和断点续跑 |
| `data-creater.py` | 人员分类训练数据采集工具。从 `data/rallies_new/` 中随机抽帧，存入 `data/person_sorter/image/` 供标注 |
| `dataset_splitter.py` | 数据集 train/val 划分工具。将 `data/person_sorter/` 中的图片按比例随机划分为训练集和验证集 |
| `yolo-train.py` | YOLO 人员分类模型训练脚本。基于 `data/person_sorter/` 数据集微调 YOLO 模型 |
| `inference_viewer.py` | 人员分类模型推理可视化工具。在 `data/rallies_new/` 上运行人员检测并可视化结果 |
| `src/utils/prepare_train_dataset.py` | 从 `rallies_annotated/` 复制训练所需文件到 `rallies_train/`（raw_clip.mp4、pose_data.json、annotations.json），支持断点续跑 |
| `src/utils/merge_annotating_data.py` | 将 `rallies_annotating/` 中新标注数据合并进 `rallies_annotated/`。转换 tracking_data.json → pose_data.json（含 court 字段），从 rally_127 续编 |
| `src/utils/add_court_to_pose.py` | 对旧数据（rallies_annotated/）补充球场关键点。逐帧跑 court 模型，将 14 点写入 pose_data.json 的 court 字段，支持断点续跑 |
| `src/utils/rerun_pose_detection.py` | 在 player1/player2 裁剪图上重跑 pose 检测（阈值 0.1），坐标映射回原始帧后用 person bbox 过滤，统计空检测帧写入 logs/pose_rerun_stats.json |

---

## 二、模型权重 (`models/`)

| 路径 | 用途 |
| --- | --- |
| `models/yolo/yolo11x-pose.pt` | YOLO11x 姿态估计模型（主力，精度最高） |
| `models/yolo/yolo26x-pose.pt` | YOLO26x 姿态估计模型（备用） |
| `models/yolo/yolov8n-pose.pt` | YOLOv8n 姿态模型（球场训练基座） |
| `models/yolo/yolo26n.pt` | YOLO26n 检测模型 |
| `models/yolo/yolo26x.pt` | YOLO26x 检测模型 |
| `models/yolo/yoloe-26l-seg.pt` | YOLOe 26L 分割模型 |
| `models/yolo/yoloe-26x-seg.pt` | YOLOe 26X 分割模型 |
| `models/court/best.pt` | 球场14点关键点检测最优权重（YOLO 微调） |
| `models/person/best.pt` | 人员分类最优权重（近端/远端球员） |
| `models/action/` | 动作识别权重（训练后产出，当前为空，旧权重归档至 `_archive/models/action_backup_20260424/`） |

---

## 三、数据集 (`data/`)

| 路径 | 内容 | 说明 |
| --- | --- | --- |
| `data/rallies_annotated/` | 199个回合（rally_001~104 + rally_127~221），全部含 `annotations.json` | 人工标注回合，用于动作识别模型训练和评估。每个回合含 `raw_clip.mp4`、`pose_data.json`（含 court 字段）、`annotations.json`（5类动作时间段标注） |
| `data/rallies_train/` | 192个回合 | 从 rallies_annotated/ 复制的训练数据（raw_clip.mp4 + pose_data.json + annotations.json） |
| `data/rallies_annotating/` | 标注工作区 | `action_annotator.py` 从 `rallies_new` 提取片段后的暂存目录，按视频来源分子文件夹。含 `_progress.json`（记录已删除片段，防止重复提取） |
| `data/court_finetune/` | 图片 + YOLO 标签 + bad_cases/ | 球场14点关键点微调数据集，含 train/val 划分 |
| `data/person_sorter/` | 图片 + YOLO 标签 | 人员分类（近端/远端球员）训练数据 |

### annotations.json 格式

```json
[
  {"start_time": 0.0, "end_time": 4.837, "action_name": "待机", "action_id": 0},
  {"start_time": 4.837, "end_time": 12.78, "action_name": "发球", "action_id": 3}
]
```

动作类别：`待机(0)` `正手(1)` `反手(2)` `发球(3)` `移动(4)`

### pose_data.json 格式

```json
[
  {
    "frame": 0,
    "court": [[x, y, conf], ...],
    "near_player": {"bbox": [x1, y1, x2, y2], "keypoints": [[x, y, conf], ...]},
    "far_player":  {"bbox": [x1, y1, x2, y2], "keypoints": [[x, y, conf], ...]}
  }
]
```

- `court`：14 个球场关键点，conf < 0.3 时该点在特征向量中置零
- `near_player` / `far_player`：17 个 COCO 骨架关键点，在 person bbox 内检测（低阈值 0.1）
- `_pose_rerun: true`：标记该帧已经过 rerun_pose_detection.py 处理

---

## 四、其他目录

| 路径 | 内容 |
| --- | --- |
| `videos/` | 原始比赛视频（25GB，MP4 + ASS 字幕），10场比赛 |
| `configs/` | YAML 训练配置，见下方详细说明 |
| `runs/court_finetune/` | 球场模型训练记录（含各版本 `weights/best.pt`） |
| `runs/yolo/` | YOLO 检测/姿态训练记录 |
| `results/` | 分析结果图表（混淆矩阵）、演示视频（`output_god_mode.mp4` 等） |
| `logs/` | 流水线运行日志（进度、错误、统计） |

---

## 四点五、训练配置文件 (`configs/`)

### 球场 / 人员分类（供 YOLO 训练使用）

| 文件 | 用途 |
| --- | --- |
| `court_keypoints.yaml` | 球场关键点检测第一版数据集配置。指向旧版 `Court_Finetune_Workspace/dataset`，4角点标注，已被后续版本替代 |
| `court_keypoints_weighted.yaml` | 球场关键点加权数据集配置。在第一版基础上合并了更多标注数据，用于第二轮微调 |
| `court_keypoints_ultimate.yaml` | 球场关键点终极版数据集配置。合并全部标注轮次数据，用于最终生产模型训练 |
| `court_14pts_weighted.yaml` | 球场14关键点加权数据集配置。升级为14点标注格式（4角点 + 10辅助点），指向 `dataset_v2`，当前主力球场训练配置 |
| `person_sorter_dataset.yaml` | 人员分类数据集配置。2类：`player_near`（近端球员）/ `player_far`（远端球员），数据在 `data/person_sorter/`，由 `yolo-train.py` 使用 |

### MSTFormer 动作识别配置（session18 重组后）

所有配置统一基准：`embed_dim=128`、`depth=8`、`use_pos_encoding=false`（关闭位置编码）。

**主配置**

| 文件 | 说明 |
| --- | --- |
| `main.yaml` | 当前最优基准。三路视觉合并（`merge_visual_tokens=true`）+ 姿态 + Focal Loss，直接用于正式训练 |

**hyperparams/ — 超参调优**

| 文件 | 变量 | 说明 |
| --- | --- | --- |
| `hp_embed96.yaml` | `embed_dim=96` | 缩小嵌入维度 |
| `hp_embed256.yaml` | `embed_dim=256` | 扩大嵌入维度 |
| `hp_depth4.yaml` | `depth=4` | 浅层 Transformer |
| `hp_depth12.yaml` | `depth=12` | 深层 Transformer |
| `hp_vtokens8.yaml` | `visual_tokens=8` | 更强视觉压缩 |
| `hp_vtokens32.yaml` | `visual_tokens=32` | 更多视觉细节 |

**ablation/ — 消融实验**

| 文件 | 变量 | 说明 |
| --- | --- | --- |
| `abl_no_pose.yaml` | `use_pose=false` | 去掉姿态输入 |
| `abl_no_crops.yaml` | `use_player_crops=false` | 去掉球员裁剪图 |
| `abl_no_visual.yaml` | `use_visual=false` | 纯姿态，无视觉流 |
| `abl_global_only.yaml` | `use_pose=false` + `use_player_crops=false` | 仅全帧视觉 |

**components/ — 组件对比**

| 文件 | 变量 | 说明 |
| --- | --- | --- |
| `cmp_focal_loss.yaml` | `loss=focal` | Focal Loss 基准 |
| `cmp_ce_loss.yaml` | `loss=cross_entropy` | CE Loss 对比 |
| `cmp_no_merge.yaml` | `merge_visual_tokens=false` | 独立三路 token（序列长 5880） |
| `cmp_resnet_backbone.yaml` | `visual_backbone=resnet18` | ResNet18 骨干（ImageNet 预训练） |
| `cmp_frozen_backbone.yaml` | `unfreeze_backbone=false` | 冻结骨干，只训练 Transformer |

旧配置（`mst_v2_*.yaml` 11个）已归档至 `_archive/configs_backup_20260424/`。

---

| 路径 | 内容 | 说明 |
| --- | --- | --- |
| `_archive/legacy_src/Hough.py` | 霍夫变换球场检测 | 早期用霍夫直线检测球场边界的实验代码 |
| `_archive/legacy_src/auto_extract_and_label.py` | 自动提取标注 | 旧版自动提取帧并预标注的脚本 |
| `_archive/legacy_src/convert_videos.py` | 视频格式转换 | 批量转换视频格式的工具 |
| `_archive/legacy_src/court_filter_test.py` | 球场过滤测试 | 球场检测过滤逻辑的测试脚本 |
| `_archive/legacy_src/demo_video_court.py` | 球场检测演示 | 早期球场检测效果演示脚本 |
| `_archive/legacy_src/action_annotator_20260423.py` | 旧版标注工具 | 2026-04-23 前的版本，读 rallies_annotated 平铺目录，无轮换提取和删除持久化 |
| `_archive/legacy_src/text.py` | 姿态提取调试 | 调试姿态关键点提取的临时脚本 |
| `_archive/legacy_src/upgrade_4_to_14.py` | 4点转14点工具 | 将旧版4角点标注转换为新版14关键点格式的迁移工具 |
| `_archive/trainData_backup_20260424/` | rallies_train 存档（104回合） | 2026-04-24 存档，特征扩展前的训练数据 |
| `_archive/unannotated_rallies/` | 无动作标注的回合（22个，rally_105~126） | 有 pose_data.json 但缺 annotations.json，暂存待后续补标 |
| `_archive/Second_Train_Dataset/` | 第二批标注数据 | 用于球场模型第二轮微调的原始标注数据 |

---

## 六、模块依赖关系

```
main.py
  ├── config_legacy.py      (路径/参数配置)
  ├── court_detector.py     (球场检测)
  └── pose_tracker.py       (姿态追踪)

train_court_pipeline.py
  └── data/court_finetune/  (训练数据)
  └── configs/*.yaml        (训练配置)

src/model/mst/train.py
  ├── model_main.py         (模型定义)
  ├── action_head.py        (分类头)
  ├── pos_encoding.py       (位置编码)
  ├── dataset.py            (数据加载)
  └── config.py             (超参数)

src/pipeline/offline_tennis_tracker.py
  └── models/yolo/          (YOLO 模型)
  └── runs/court_finetune/  (球场模型)

src/utils/action_annotator.py
  └── data/rallies_new/         (源数据，按视频来源分子文件夹)
  └── data/rallies_annotating/  (工作区，含 _progress.json 进度记录)
```

---

## 七、标注工作流

```
1. 采集视频
   videos/ 中放入原始比赛视频

2. 提取回合
   main.py → data/rallies_new/{比赛名}/rally_xxx/

3. 标注球场关键点（用于球场模型微调）
   smart_extract_14pts.py → 预标注帧
   corner_driven_refine_tool.py → 手动精修
   prepare_weighted_dataset.py → 合并到 data/court_finetune/
   train_court_pipeline.py → 训练新模型

4. 标注球员动作（用于动作识别模型训练）
   action_annotator.py → 从 rallies_new 按来源轮换提取片段到 rallies_annotating/
   在浏览器中标注 annotations.json（支持删除片段、进度持久化）
   src/model/train.py → 训练 MSTFormer

5. 标注人员分类（用于近/远端球员识别）
   data-creater.py → 采样图片
   label_tool.py → 标注 bounding box
   dataset_splitter.py → 划分 train/val
   yolo-train.py → 训练分类模型
```
