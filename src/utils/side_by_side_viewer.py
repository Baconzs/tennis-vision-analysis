"""side_by_side_viewer.py — 双视图对比播放器

功能：并排显示原始视频与追踪结果视频，用于直观对比效果
"""
import cv2
import json
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from collections import defaultdict, deque

# =====================================================================
# 1. 全局配置与物理映射常量
# =====================================================================
# 注意：路径前面加了 r，完美解决 \c 转义报错
DATASET_ROOT = "data/rallies_annotated"

COURT_PHYSICAL = np.array([
    [-5.485, -11.885], [5.485, -11.885], [5.485, 11.885], [-5.485, 11.885],
    [0.000, -11.885], [0.000, 11.885], [-4.115, -6.400], [4.115, -6.400],
    [0.000, -6.400], [-4.115, 6.400], [4.115, 6.400], [0.000, 6.400],
    [-5.485, 0.000], [5.485, 0.000]
], dtype=np.float32)

COURT_LINES = [(0, 1), (2, 3), (0, 3), (1, 2), (6, 7), (9, 10), (4, 5), (12, 13)]

POSE_PAIRS = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
              (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10)]


# =====================================================================
# 2. 核心数学工具
# =====================================================================
def get_homography(court_data):
    if not court_data: return None
    kpts = np.array(court_data)
    mask = kpts[:, 2] > 0.4
    if np.sum(mask) < 4: return None

    phys_pts = COURT_PHYSICAL[mask]
    pixel_pts = kpts[mask, :2]
    weights = kpts[mask, 2]

    H_init, _ = cv2.findHomography(phys_pts, pixel_pts, cv2.RANSAC, 5.0)
    if H_init is None: return None

    def residuals(h):
        H = np.append(h, 1.0).reshape(3, 3)
        pts_3d = np.concatenate([phys_pts, np.ones((len(phys_pts), 1))], axis=1)
        proj = (H @ pts_3d.T).T
        proj[:, :2] /= (proj[:, 2:] + 1e-8)
        return ((proj[:, :2] - pixel_pts) * weights[:, np.newaxis]).flatten()

    res = least_squares(residuals, x0=(H_init / H_init[2, 2]).flatten()[:8], method='lm')
    return np.append(res.x, 1.0).reshape(3, 3)


def extract_feet_pos(player):
    bx = player["bbox"]
    feet_px = np.array([(bx[0] + bx[2]) / 2, bx[3]])  # 兜底逻辑
    pose = player.get("pose")
    if pose:
        kp = np.array(pose)
        if len(kp) >= 17:
            l_a, r_a, l_k, r_k = kp[15], kp[16], kp[13], kp[14]
            if l_a[2] > 0.35 and r_a[2] > 0.35:
                feet_px = (l_a[:2] + r_a[:2]) / 2.0
            elif l_a[2] > 0.35:
                feet_px = l_a[:2]
            elif r_a[2] > 0.35:
                feet_px = r_a[:2]
            elif l_k[2] > 0.4 and r_k[2] > 0.4:
                feet_px = (l_k[:2] + r_k[:2]) / 2.0
    return feet_px


# =====================================================================
# 3. 审查工作台 GUI 引擎
# =====================================================================
class ReviewDashboard:
    def __init__(self, json_files):
        self.files = json_files
        self.current_idx = 0
        self.cap = None
        self.total_frames = 1
        self.current_frame = 0
        self.frames_data = {}

        # 交互状态
        self.is_paused = False
        self.need_video_reload = True
        self.list_scroll = 0
        self.is_dragging = False
        self.last_H = None
        self.radar_history = defaultdict(lambda: deque(maxlen=30))

        # UI 尺寸布局 (总尺寸 1624 x 720)
        self.W_LIST = 250
        self.W_VID = 1024
        self.H_VID = 576
        self.W_RADAR = 350
        self.H_TLINE = 144
        self.H_TOTAL = self.H_VID + self.H_TLINE
        self.W_TOTAL = self.W_LIST + self.W_VID + self.W_RADAR

        self.window_name = "Tennis God Mode Dashboard"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1624, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_event)

    def load_video(self, idx):
        if self.cap: self.cap.release()
        json_path = self.files[idx]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.frames_data = {f["frame_id"]: f for f in data["frames"]}

        video_path = json_path.parent / "raw_clip.mp4"
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.current_frame = 0
        self.last_H = None
        self.radar_history.clear()
        self.need_video_reload = False

    def mouse_event(self, event, x, y, flags, param):
        """核心交互：处理所有的鼠标点击、拖拽和滚轮滑动"""
        # 1. 列表区域交互 (x < 250)
        if x < self.W_LIST:
            if event == cv2.EVENT_MOUSEWHEEL:
                if flags > 0:
                    self.list_scroll = max(0, self.list_scroll - 1)
                else:
                    self.list_scroll = min(len(self.files) - 1, self.list_scroll + 1)
            elif event == cv2.EVENT_LBUTTONDOWN:
                click_idx = self.list_scroll + (y - 50) // 35
                if 0 <= click_idx < len(self.files):
                    self.current_idx = click_idx
                    self.need_video_reload = True

        # 2. 进度条区域交互 (中下方区域)
        elif self.W_LIST <= x <= self.W_LIST + self.W_VID and y > self.H_VID:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.is_dragging = True
                self.set_frame_by_mouse(x)
            elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
                self.set_frame_by_mouse(x)
            elif event == cv2.EVENT_LBUTTONUP:
                self.is_dragging = False

    def set_frame_by_mouse(self, x):
        """将鼠标的 X 坐标映射为视频帧数"""
        bar_x = x - self.W_LIST - 50  # 减去边距
        bar_w = self.W_VID - 100
        progress = np.clip(bar_x / bar_w, 0, 1)
        self.current_frame = int(progress * (self.total_frames - 1))
        if self.cap: self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def draw_list(self, canvas):
        """绘制左侧播放列表"""
        canvas[0:self.H_TOTAL, 0:self.W_LIST] = (30, 35, 30)
        cv2.putText(canvas, "PLAYLIST", (20, 35), 0, 0.8, (255, 255, 255), 2)
        cv2.line(canvas, (10, 45), (self.W_LIST - 10, 45), (100, 100, 100), 1)

        for i in range(20):  # 最多显示 20 项
            idx = self.list_scroll + i
            if idx >= len(self.files): break

            y_pos = 75 + i * 35
            name = self.files[idx].parent.name[:20]  # 截断过长名字
            color = (200, 200, 200)

            # 高亮当前选中的视频
            if idx == self.current_idx:
                cv2.rectangle(canvas, (5, y_pos - 25), (self.W_LIST - 5, y_pos + 5), (80, 120, 80), -1)
                color = (255, 255, 255)

            cv2.putText(canvas, f"{idx + 1}. {name}", (15, y_pos - 5), 0, 0.5, color, 1)

    def draw_timeline(self, canvas):
        """绘制下方时间轴面板"""
        t_y = self.H_VID
        canvas[t_y:self.H_TOTAL, self.W_LIST:self.W_LIST + self.W_VID] = (40, 40, 45)

        # 播放状态与提示文字
        state_str = "PAUSED" if self.is_paused else "PLAYING"
        color_state = (0, 0, 255) if self.is_paused else (0, 255, 0)
        cv2.putText(canvas, f"STATUS: {state_str}", (self.W_LIST + 50, t_y + 40), 0, 0.7, color_state, 2)
        cv2.putText(canvas, f"Frame: {self.current_frame}/{self.total_frames}", (self.W_LIST + 250, t_y + 40), 0, 0.7,
                    (200, 200, 200), 2)
        cv2.putText(canvas, "Controls: [SPACE] Play/Pause | [A/D] Step Frame | [Mouse] Drag Timeline",
                    (self.W_LIST + 50, t_y + 120), 0, 0.5, (150, 150, 150), 1)

        # 进度条轨道与滑块
        bar_y = t_y + 70
        bar_w = self.W_VID - 100
        bar_start_x = self.W_LIST + 50

        cv2.line(canvas, (bar_start_x, bar_y), (bar_start_x + bar_w, bar_y), (100, 100, 100), 4)
        progress = self.current_frame / max(1, self.total_frames - 1)
        current_x = int(bar_start_x + progress * bar_w)
        cv2.line(canvas, (bar_start_x, bar_y), (current_x, bar_y), (0, 255, 255), 4)
        cv2.circle(canvas, (current_x, bar_y), 10, (255, 255, 255), -1)

    def run(self):
        while True:
            if self.need_video_reload: self.load_video(self.current_idx)

            if not self.is_paused and not self.is_dragging:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.radar_history.clear()
                    continue
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret: frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            # 准备全局画布
            canvas = np.zeros((self.H_TOTAL, self.W_TOTAL, 3), dtype=np.uint8)
            f_data = self.frames_data.get(self.current_frame, {})

            # ================= 渲染主视角 =================
            v_frame = cv2.resize(frame, (self.W_VID, self.H_VID))

            # 缩放因子：因为画面被 resize 了，画线的坐标也得跟着缩放
            sx, sy = self.W_VID / frame.shape[1], self.H_VID / frame.shape[0]

            court = f_data.get("court")
            H = get_homography(court)
            if H is not None: self.last_H = H

            # 画球场 (修复 NoneType 报错)
            if court is not None:
                for l in COURT_LINES:
                    if l[0] < len(court) and l[1] < len(court):
                        p1, p2 = court[l[0]], court[l[1]]
                        if p1[2] > 0.4 and p2[2] > 0.4:
                            cv2.line(v_frame, (int(p1[0] * sx), int(p1[1] * sy)),
                                     (int(p2[0] * sx), int(p2[1] * sy)), (0, 255, 0), 2)

            for player in f_data.get("players", []):
                bx = player["bbox"]
                cv2.rectangle(v_frame, (int(bx[0] * sx), int(bx[1] * sy)),
                              (int(bx[2] * sx), int(bx[3] * sy)), (0, 255, 255), 2)

                pose = player.get("pose")
                if pose:
                    for p1_i, p2_i in POSE_PAIRS:
                        if p1_i < len(pose) and p2_i < len(pose):
                            p1, p2 = pose[p1_i], pose[p2_i]
                            if p1[2] > 0.3 and p2[2] > 0.3:
                                cv2.line(v_frame, (int(p1[0] * sx), int(p1[1] * sy)),
                                         (int(p2[0] * sx), int(p2[1] * sy)), (0, 0, 255), 2)

                # 映射到雷达
                if self.last_H is not None:
                    feet = extract_feet_pos(player)
                    pt = np.array([[[feet[0], feet[1]]]], dtype=np.float32)
                    real_pos = cv2.perspectiveTransform(pt, np.linalg.inv(self.last_H))[0][0]
                    self.radar_history[player["id"]].append(real_pos)

            canvas[0:self.H_VID, self.W_LIST:self.W_LIST + self.W_VID] = v_frame

            # ================= 渲染扩容版上帝视角 =================
            radar = np.zeros((self.H_TOTAL, self.W_RADAR, 3), dtype=np.uint8)
            cv2.rectangle(radar, (0, 0), (self.W_RADAR, self.H_TOTAL), (35, 55, 35), -1)

            # 雷达 Y 轴放到了极大的空间，完美包容退网打法
            scale = 16
            cx, cy = self.W_RADAR // 2, self.H_TOTAL // 2 - 30
            hw, hh = 5.485 * scale, 11.885 * scale
            cv2.rectangle(radar, (int(cx - hw), int(cy - hh)), (int(cx + hw), int(cy + hh)), (255, 255, 255), 2)
            cv2.line(radar, (int(cx - hw), cy), (int(cx + hw), cy), (255, 255, 255), 2)
            cv2.putText(radar, "Expanded Baseline Runoff", (10, 30), 0, 0.6, (150, 150, 150), 1)

            # 绘制历史拖尾与当前点
            for pid, hist in self.radar_history.items():
                pts = list(hist)
                for i in range(1, len(pts)):
                    px1 = (int(pts[i - 1][0] * scale + cx), int(-pts[i - 1][1] * scale + cy))
                    px2 = (int(pts[i][0] * scale + cx), int(-pts[i][1] * scale + cy))
                    alpha = i / len(pts)
                    cv2.line(radar, px1, px2, (0, int(200 * alpha), int(255 * alpha)), max(1, int(3 * alpha)))
                if pts:
                    cur_px = (int(pts[-1][0] * scale + cx), int(-pts[-1][1] * scale + cy))
                    cv2.circle(radar, cur_px, 6, (0, 255, 255), -1)

            canvas[0:self.H_TOTAL, self.W_LIST + self.W_VID:self.W_TOTAL] = radar

            # ================= 渲染 UI 组件 =================
            self.draw_list(canvas)
            self.draw_timeline(canvas)

            cv2.imshow(self.window_name, canvas)

            # ================= 键盘监听 =================
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == 32:
                self.is_paused = not self.is_paused
            elif key == ord('d') and self.is_paused:
                self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
            elif key == ord('a') and self.is_paused:
                self.current_frame = max(0, self.current_frame - 1)

        if self.cap: self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    base_path = Path(DATASET_ROOT)
    json_files = list(base_path.rglob("tracking_data.json"))
    if not json_files:
        print("未找到 JSON 数据文件。")
    else:
        app = ReviewDashboard(json_files)
        app.run()