# model/yolo/ — 单帧 YOLO 动作分类（对比基线）

把动作识别当作**单帧图像分类**问题的基线模型：YOLO11n backbone + 全局平均池化 + 分类头，输出 5 类动作。用于和时序模型 MSTFormer 做对比，说明「逐帧分类」相对「时序建模」的不足。

| 文件 | 作用 |
| --- | --- |
| `model.py` | 模型定义 `YoloFrameClassifier`：hook 取 YOLO backbone 最后输出，接分类头 |
| `dataset.py` | 单帧数据集：从回合帧 + `annotations.json` 取每帧的动作标签 |
| `train.py` | 训练脚本 |

## 运行

```bash
python src/model/yolo/train.py
```

> 与 `model/mst/` 共享同一套 `annotations.json` 标注和动作类别定义（待机/正手/反手/发球/移动）。
