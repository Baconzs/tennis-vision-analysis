"""player.py — 视频播放器，基于 QTimer + OpenCV 逐帧读取"""
import ctypes
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, pyqtSignal, QObject


def _get_short_path(path: str) -> str:
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # 非 Windows 直接用原路径
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


class VideoPlayer(QObject):
    frame_changed = pyqtSignal(int, object)   # (frame_idx, rgb_ndarray)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._fps = 30.0
        self._total = 0
        self._current = 0
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

    def load(self, video_path: str) -> bool:
        if self._cap:
            self._cap.release()
        short = _get_short_path(video_path)
        cap = cv2.VideoCapture(short)
        if not cap.isOpened():
            return False
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps > 0 and not np.isnan(fps) else 30.0
        self._total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current = 0
        return True

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total

    @property
    def current_frame(self) -> int:
        return self._current

    def seek(self, frame_idx: int):
        if self._cap is None:
            return
        frame_idx = max(0, min(frame_idx, self._total - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._current = frame_idx
        self._emit_current()

    def play(self):
        if self._cap is None or self._playing:
            return
        self._playing = True
        interval = max(1, int(1000 / self._fps))
        self._timer.start(interval)

    def pause(self):
        self._playing = False
        self._timer.stop()

    def toggle(self):
        if self._playing:
            self.pause()
        else:
            self.play()

    def step(self, delta: int):
        self.seek(self._current + delta)

    def _next_frame(self):
        if self._cap is None:
            return
        ret, frame = self._cap.read()
        if not ret:
            self.pause()
            self.finished.emit()
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_changed.emit(self._current, rgb)
        self._current += 1

    def _emit_current(self):
        if self._cap is None:
            return
        ret, frame = self._cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_changed.emit(self._current, rgb)
            # 回退一帧，保持位置不变
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current)

    def release(self):
        self.pause()
        if self._cap:
            self._cap.release()
            self._cap = None
