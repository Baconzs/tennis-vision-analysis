"""timeline.py — 时间轴组件：GT条 / 预测条 / 帧格子条"""
from PyQt5.QtWidgets import QWidget, QScrollArea, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QColor, QFont, QPen

ACTION_COLORS = [
    QColor("#607D8B"),  # 0 待机
    QColor("#4CAF50"),  # 1 正手
    QColor("#2196F3"),  # 2 反手
    QColor("#FF9800"),  # 3 发球
    QColor("#9C27B0"),  # 4 移动
]
ACTION_NAMES = ["待机", "正手", "反手", "发球", "移动"]
COLOR_UNKNOWN = QColor("#37474F")
COLOR_BG = QColor("#1E1E2E")
COLOR_CURSOR = QColor("#FFFFFF")


def _action_color(action_id):
    if 0 <= action_id < len(ACTION_COLORS):
        return ACTION_COLORS[action_id]
    return COLOR_UNKNOWN


class ActionBarWidget(QWidget):
    """GT 标注条或模型预测条，按时间比例绘制彩色区间块。"""

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self._label = label
        self._total_frames = 0
        self._fps = 30.0
        self._segments = []   # [{start, end, action_id}]
        self._cursor = 0
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)

    def set_data(self, total_frames, fps, segments):
        self._total_frames = total_frames
        self._fps = fps
        self._segments = segments
        self.update()

    def set_cursor(self, frame_idx):
        self._cursor = frame_idx
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, COLOR_BG)

        if self._total_frames == 0:
            p.setPen(QColor("#555"))
            p.setFont(QFont("Microsoft YaHei", 8))
            p.drawText(QRect(0, 0, w, h), Qt.AlignCenter, f"{self._label}（未加载）")
            return

        for seg in self._segments:
            x1 = int(seg["start"] / self._total_frames * w)
            x2 = int(seg["end"] / self._total_frames * w)
            p.fillRect(x1, 2, max(x2 - x1, 2), h - 4, _action_color(seg["action_id"]))

        p.setFont(QFont("Microsoft YaHei", 7))
        p.setPen(QColor("#CCCCCC"))
        p.drawText(QRect(4, 0, 60, h), Qt.AlignVCenter | Qt.AlignLeft, self._label)

        cx = int(self._cursor / max(self._total_frames, 1) * w)
        p.setPen(QPen(COLOR_CURSOR, 1))
        p.drawLine(cx, 0, cx, h)

    def mouseMoveEvent(self, event):
        if self._total_frames == 0:
            return
        frame = int(event.x() / self.width() * self._total_frames)
        for seg in self._segments:
            if seg["start"] <= frame <= seg["end"]:
                name = ACTION_NAMES[seg["action_id"]] if 0 <= seg["action_id"] < len(ACTION_NAMES) else "未知"
                self.setToolTip(
                    f"{self._label}：{name}\n"
                    f"帧 {seg['start']}–{seg['end']}  "
                    f"({seg['start']/self._fps:.2f}s – {seg['end']/self._fps:.2f}s)"
                )
                return
        self.setToolTip(f"帧 {frame}  ({frame/self._fps:.2f}s)")


class FrameTrackWidget(QWidget):
    """帧格子条：每格一帧，彩色，随播放滚动。"""
    CELL_W = 5
    CELL_H = 24

    def __init__(self, parent=None):
        super().__init__(parent)
        self._total_frames = 0
        self._per_frame_action = []
        self._cursor = 0
        self.setFixedHeight(self.CELL_H)

    def set_data(self, total_frames, per_frame_action):
        self._total_frames = total_frames
        self._per_frame_action = per_frame_action
        self.setFixedWidth(max(total_frames * self.CELL_W, 1))
        self.update()

    def set_cursor(self, frame_idx):
        self._cursor = frame_idx
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.CELL_H
        p.fillRect(0, 0, w, h, COLOR_BG)

        for i, action_id in enumerate(self._per_frame_action):
            x = i * self.CELL_W
            p.fillRect(x + 1, 1, self.CELL_W - 1, h - 2, _action_color(action_id))

        if 0 <= self._cursor < self._total_frames:
            cx = self._cursor * self.CELL_W
            p.setPen(QPen(COLOR_CURSOR, 1))
            p.drawRect(cx, 0, self.CELL_W - 1, h - 1)


class TimelinePanel(QWidget):
    """完整时间轴面板：GT条 + 预测条 + 帧格子条（带横向滚动）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(3)

        self.gt_bar = ActionBarWidget("GT 标注")
        layout.addWidget(self.gt_bar)

        self.pred_bar = ActionBarWidget("模型预测")
        layout.addWidget(self.pred_bar)

        lbl = QLabel("帧轨道")
        lbl.setFont(QFont("Microsoft YaHei", 8))
        lbl.setStyleSheet("color: #888; padding-left: 4px;")
        layout.addWidget(lbl)

        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setFixedHeight(FrameTrackWidget.CELL_H + 20)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: #1E1E2E; }")

        self.frame_track = FrameTrackWidget()
        self._scroll.setWidget(self.frame_track)
        layout.addWidget(self._scroll)

    def load_gt(self, total_frames, fps, anno_json):
        """从 annotations.json 解析 GT 区间并渲染。"""
        segments = []
        for seg in (anno_json if isinstance(anno_json, list) else []):
            start_f = round(seg.get("start_time", 0) * fps)
            end_f = round(seg.get("end_time", 0) * fps)
            segments.append({"start": start_f, "end": end_f, "action_id": seg.get("action_id", 0)})
        self.gt_bar.set_data(total_frames, fps, segments)

    def load_predictions(self, total_frames, fps, per_frame_preds):
        """per_frame_preds: list[int]，长度 == total_frames，每帧的预测类别。"""
        # 将逐帧预测压缩成区间段
        segments = []
        if per_frame_preds:
            cur_id = per_frame_preds[0]
            cur_start = 0
            for i, pid in enumerate(per_frame_preds[1:], 1):
                if pid != cur_id:
                    segments.append({"start": cur_start, "end": i - 1, "action_id": cur_id})
                    cur_id = pid
                    cur_start = i
            segments.append({"start": cur_start, "end": len(per_frame_preds) - 1, "action_id": cur_id})

        self.pred_bar.set_data(total_frames, fps, segments)
        self.frame_track.set_data(total_frames, per_frame_preds)

    def reset_predictions(self):
        """切换 rally 时清空预测条，显示"未定义"。"""
        self.pred_bar.set_data(0, 30.0, [])
        self.frame_track.set_data(0, [])

    def reset_gt(self):
        """切换 rally 时清空 GT 条。"""
        self.gt_bar.set_data(0, 30.0, [])

    def set_cursor(self, frame_idx):
        self.gt_bar.set_cursor(frame_idx)
        self.pred_bar.set_cursor(frame_idx)
        self.frame_track.set_cursor(frame_idx)
        # 自动滚动帧格子条，使当前帧居中
        scroll_x = frame_idx * FrameTrackWidget.CELL_W - self._scroll.viewport().width() // 2
        self._scroll.horizontalScrollBar().setValue(max(0, scroll_x))
