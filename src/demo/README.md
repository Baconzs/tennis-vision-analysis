# demo/ — PyQt5 桌面可视化 Demo

视频播放 + 三行时间轴 + MSTFormer 实时推理可视化的桌面应用，用来直观查看模型在某个回合上的预测效果。

## 运行

```bash
python src/demo/main.py \
  --rally   data/rallies_annotated/rally_001_19.8s \
  --config  configs/main.yaml \
  --weights models/action/<config>/<时间戳>/best.pth \
  --person  models/person/best.pt \   # 可选：填了就实时检测
  --pose    models/yolo/yolo11x-pose.pt  # 可选
```

参数也可在界面里选。`--person`/`--pose` 两种工作模式：

- **填了**：实时跑 person 检测 + pose 估计，绘制 bbox 与骨架叠加到画面；
- **不填**：回退读取回合目录里预提取的 `pose_data.json` 和裁剪图。

两种模式都把整段序列一次性送入 MSTFormer 推理。

## 文件

| 文件 | 作用 |
| --- | --- |
| `main.py` | 入口。解析参数；处理 Windows 下 torch/PyQt5 的 CUDA DLL 与 Qt 插件加载顺序 |
| `app.py` | 主窗口：视频播放、三行时间轴、文件/模型选择、推理触发、动作图例 |
| `player.py` | 视频播放器：QTimer + OpenCV 逐帧读取，处理中文路径 |
| `timeline.py` | 三行时间轴：GT 标注条 / 预测条 / 帧格子条，游标跟随播放 |
| `inference.py` | 推理线程（QThread），两种模式见上 |
| `seq_len_sweep.py` | 序列长度扫描脚本：遍历不同 `seq_len` 输出准确率 CSV |

> 依赖 `PyQt5`，需在有图形界面的环境运行（远程服务器需 X11/VNC）。
