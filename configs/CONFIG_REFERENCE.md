# MSTFormer 配置字段参考手册

> 基于代码分析自动生成，覆盖 `model_main.py`、`backbone_factory.py`、`train.py`、`dataset.py`、`config.py`。
> 更新：2026-04-24

---

## 一、数据字段

| 字段 | 必须 | 默认值 | 作用 |
|------|------|--------|------|
| `data_root` | ✅ | — | 训练数据根目录（含各 rally 子目录） |
| `crops_root` | ✅ | — | 球员裁剪图根目录（通常与 data_root 相同） |
| `seq_len` | ✅ | — | 每个样本的序列帧数（如 120） |
| `min_seq_len` | ❌ | `max(30, seq_len//2)` | reshuffle 时切片最短帧数，仅当 `reshuffle_augment=true` 生效 |
| `train_ratio` | ✅ | — | 训练集占比（0~1） |
| `num_classes` | ✅ | — | 动作分类类别数（当前为 5） |
| `class_weights` | ✅ | — | 各类别损失权重列表，长度须等于 `num_classes` |

---

## 二、模型架构开关

| 字段 | 必须 | 默认值 | 作用 | 冲突/依赖 |
|------|------|--------|------|-----------|
| `use_visual` | ❌ | `true` | 是否启用视觉流；`false` 为纯姿态模式 | `false` 时所有视觉相关字段均无效 |
| `use_player_crops` | ❌ | `true` | 是否加载球员裁剪图（p1/p2）；`false` 时仅全帧一路 | 仅当 `use_visual=true` 生效 |
| `use_pose` | ❌ | `true` | 是否使用姿态特征；`false` 时 pose 向量置零（token 槽保留） | 无 |
| `use_pos_encoding` | ❌ | `false` | 是否启用正弦位置编码；关闭时模型不依赖绝对位置信息 | 无 |
| `keyframe_only` | ❌ | `false` | `true` 时仅训练关键帧检测头，不建 action_head | `true` 时 `keyframe_loss_weight` 无效 |
| `merge_visual_tokens` | ❌ | `false` | `true` 时三路 token cat 后过 shared_resampler 压到 `visual_tokens` 个 | 仅当 `use_visual=true` 且 `use_player_crops=true` 有意义 |
| `parallel_backbones` | ❌ | `false` | `true` 时三路骨干并行（需更多显存，单卡易 OOM） | 仅当 `use_visual=true` 且 `use_player_crops=true` 生效 |

---

## 三、视觉骨干网络

### 通用字段

| 字段 | 必须 | 默认值 | 作用 | 生效条件 |
|------|------|--------|------|----------|
| `visual_backbone` | ❌ | `"yolo11"` | 骨干类型：`yolo11` / `vit` / `resnet18` / `raw` | `use_visual=true` |
| `visual_tokens` | ❌ | `16` | 最终每路输出的视觉 token 数 | `visual_backbone="yolo11"` 时控制 TokenResampler 输出；`merge_visual_tokens=true` 时控制合并后总数；**vit 模式下无效** |

### yolo11 专属

| 字段 | 必须 | 默认值 | 作用 | 生效条件 |
|------|------|--------|------|----------|
| `backbone_weights` | ✅* | — | p1/p2 骨干权重路径（yolo11n-pose.pt） | `visual_backbone="yolo11"` |
| `global_backbone_weights` | ❌ | 同 `backbone_weights` | 全帧骨干权重路径（yolo11n.pt）；不填则与 p1/p2 共用 | `visual_backbone="yolo11"` 且 `use_player_crops=true` |
| `unfreeze_backbone` | ❌ | `false` | `true` 时解冻骨干参数并启用 gradient checkpointing | `visual_backbone="yolo11"`；**vit 模式下无效** |
| `multi_scale_levels` | ✅* | — | 多尺度特征层级，如 `[3, 4, 5]` 对应 P3/P4/P5 | `visual_backbone="yolo11"` |
| `tokens_per_scale` | ✅* | — | 每个尺度 spatial pool 后的 token 数（如 4 → 2×2） | `visual_backbone="yolo11"` |

> ✅* 表示在 `visual_backbone="yolo11"` 时为必须字段（代码硬读，缺失报 KeyError）；vit 模式下这些字段可以写占位值，不会被读取。

### vit 专属

| 字段 | 必须 | 默认值 | 作用 | 生效条件 |
|------|------|--------|------|----------|
| `vit_patch_grid` | ❌ | `4` | patch 网格大小，patch token 经 TokenResampler 压到 `visual_tokens` 个 | `visual_backbone="vit"` |

> `vit_depth` / `vit_num_heads` 已废弃（session15 重构后 ViT 内部无 Transformer，字段保留兼容性但不被读取）。

---

## 四、模型超参

| 字段 | 必须 | 默认值 | 作用 |
|------|------|--------|------|
| `embed_dim` | ✅ | — | 模型嵌入维度（所有 token 的统一维度） |
| `depth` | ✅ | — | 主 TransformerEncoder 层数 |
| `num_heads` | ✅ | — | 主 Transformer 多头注意力头数（须能整除 `embed_dim`） |
| `dropout` | ❌ | `0.1` | Transformer 层 dropout 比例 |

---

## 五、训练超参

| 字段 | 必须 | 默认值 | 作用 | 冲突/依赖 |
|------|------|--------|------|-----------|
| `batch_size` | ✅ | — | 实际批大小 | 须能整除 `virtual_batch_size` |
| `virtual_batch_size` | ✅ | — | 虚拟批大小；`accumulation_steps = virtual_batch_size / batch_size` | 须为 `batch_size` 的整数倍 |
| `total_epochs` | ✅ | — | 总训练轮数 | |
| `learning_rate` | ✅ | — | 初始学习率（AdamW） | |
| `weight_decay` | ✅ | — | AdamW 权重衰减 | |
| `warmup_epochs` | ❌ | `5` | 线性 warmup 轮数，之后余弦退火到 `lr × 0.01` | |
| `loss` | ❌ | `"cross_entropy"` | 损失函数：`cross_entropy` 或 `focal` | |
| `focal_gamma` | ❌ | `2.0` | Focal Loss 的 γ 参数 | 仅当 `loss="focal"` 生效 |
| `keyframe_loss_weight` | ❌ | `0.5` | 关键帧 loss 权重：`total = loss_action + weight × loss_kf` | 仅当 `keyframe_only=false` 生效 |

---

## 六、硬件与数据加载

| 字段 | 必须 | 默认值 | 作用 |
|------|------|--------|------|
| `num_workers` | ✅ | — | DataLoader 工作进程数（Windows 建议 2） |
| `pin_memory` | ✅ | — | 是否锁定内存加速 GPU 传输 |
| `reshuffle_augment` | ❌ | `true` | `true` 时每 epoch 随机重划训练切片（时序增强） |
| `image_augment` | ❌ | `false` | `true` 时训练集开启图像级增强：颜色抖动/高斯噪声/模糊/随机擦除/半透明覆盖 |
| `transformer_checkpoint` | ❌ | `true` | `true` 时 Transformer 层启用 gradient checkpointing，省显存约 50% |

---

## 七、自动生成字段（勿手动填写）

以下字段由 `config.py` 或 `train.py` 在运行时自动注入，不应出现在 YAML 中：

| 字段 | 来源 | 说明 |
|------|------|------|
| `device` | `config.py` | 自动检测 cuda / cpu |
| `accumulation_steps` | `config.py` | `virtual_batch_size / batch_size` |
| `_yaml_path` | `train.py` | 配置文件路径，用于日志记录 |
| `_smoke_clip` | `train.py` | `--smoke` 模式下的测试 clip 路径 |

---

## 八、各配置文件字段有效性速查

### 主配置

| 字段 | main |
|------|:----:|
| `use_visual` | ✅ |
| `use_player_crops` | ✅ |
| `use_pose` | ✅ |
| `use_pos_encoding` | ❌ false |
| `merge_visual_tokens` | ✅ true |
| `unfreeze_backbone` | ✅ |

### ablation/

| 字段 | abl_no_pose | abl_no_crops | abl_no_visual | abl_global_only |
|------|:-----------:|:------------:|:-------------:|:---------------:|
| `use_visual` | ✅ | ✅ | ❌ false | ✅ |
| `use_player_crops` | ✅ | ❌ false | ❌ false | ❌ false |
| `use_pose` | ❌ false（置零） | ✅ | ✅ | ❌ false（置零） |

### components/

| 字段 | cmp_focal_loss | cmp_ce_loss | cmp_no_merge | cmp_resnet_backbone | cmp_frozen_backbone |
|------|:--------------:|:-----------:|:------------:|:-------------------:|:-------------------:|
| `loss` | focal | cross_entropy | focal | focal | focal |
| `merge_visual_tokens` | ✅ true | ✅ true | ❌ false | ✅ true | ✅ true |
| `visual_backbone` | yolo11 | yolo11 | yolo11 | resnet18 | yolo11 |
| `unfreeze_backbone` | ✅ | ✅ | ✅ | ✅ | ❌ false |

> "占位"：字段存在但代码不读取（因走了不同分支），删除会报 KeyError；"无效"：字段不存在或存在均不影响结果。

---

## 九、常见配置错误

| 错误 | 后果 |
|------|------|
| `num_heads` 不能整除 `embed_dim` | 运行时报错 |
| `batch_size` 不能整除 `virtual_batch_size` | 梯度累积步数为小数，训练行为异常 |
| `vit` 模式下设置 `visual_tokens` | 无效，token 数由 `vit_patch_grid²` 固定 |
| `merge_visual_tokens=true` 但 `use_player_crops=false` | shared_resampler 只处理一路，合并无意义 |
| `loss="cross_entropy"` 时设置 `focal_gamma` | 无效，不影响运行但容易误导 |
| `keyframe_only=true` 时设置 `keyframe_loss_weight` | 无效，只有关键帧 loss，无动作 loss 可加权 |
