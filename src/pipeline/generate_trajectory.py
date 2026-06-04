"""generate_trajectory.py — 球员运动轨迹生成模块

功能：从追踪结果中提取球员坐标序列，生成用于动作识别的轨迹特征
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import least_squares

# =====================================================================
# 1. 全局配置区
# =====================================================================
VIDEO_PATH = "your_video.mp4"  # 【修改这里】测试视频路径
COURT_MODEL_PATH = "court_model.pt"  # 【修改这里】14点场地模型路径
POSE_MODEL_PATH = "yolo11x-pose.pt"  # 【修改这里】肢体模型路径
OUTPUT_PATH = "output_radar_ultimate.mp4"  # 输出视频路径

# 物理坐标库 (14个关键点，以球网中心为原点)
COURT_14_PTS_PHYSICAL = np.array([
    [-5.485, -11.885], [5.485, -11.885], [5.485, 11.885], [-5.485, 11.885],  # 0-3: 核心外角点
    [0.000, -11.885], [0.000, 11.885],  # 4-5: 底线中点
    [-4.115, -6.400], [4.115, -6.400], [0.000, -6.400],  # 发球线与单打边线/中线交点
    [-4.115, 6.400], [4.115, 6.400], [0.000, 6.400],  # 发球线与单打边线/中线交点
    [-5.485, 0.000], [5.485, 0.000]  # 球网与双打边线交点
], dtype=np.float32)

# 先验权重体系: 核心角点拉扯力强，内部线段拉扯力弱
BASE_WEIGHTS = np.array([7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.float32)


# =====================================================================
# 2. 数学优化引擎：SciPy 加权最小二乘法 (L-M)
# =====================================================================
def reprojection_residuals(h_elements, src_pts, dst_pts, weights):
    """计算加权投影误差的残差函数"""
    H = np.append(h_elements, 1.0).reshape(3, 3)
    src_pts_3d = np.concatenate([src_pts, np.ones((len(src_pts), 1))], axis=1)
    proj_pts_3d = (H @ src_pts_3d.T).T

    # 防止除以 0
    proj_pts_3d[:, 2] = np.where(proj_pts_3d[:, 2] == 0, 1e-7, proj_pts_3d[:, 2])
    proj_pts_2d = proj_pts_3d[:, :2] / proj_pts_3d[:, 2:]

    errors = proj_pts_2d - dst_pts
    weighted_errors = errors * weights[:, np.newaxis]
    return weighted_errors.flatten()


def get_weighted_homography(phys_pts, pixel_pts, weights):
    """使用加权 L-M 算法计算精确的单应性矩阵"""
    # 先用 OpenCV 算一个粗略的初值
    H_init, _ = cv2.findHomography(phys_pts, pixel_pts, cv2.RANSAC, 5.0)
    if H_init is None: return None

    h_initial_guess = (H_init / H_init[2, 2]).flatten()[:8]
    res = least_squares(
        reprojection_residuals, x0=h_initial_guess,
        args=(phys_pts, pixel_pts, weights), method='lm'
    )
    return np.append(res.x, 1.0).reshape(3, 3)


# =====================================================================
# 3. 双重滑动窗口滤波器 (矩阵 + 轨迹)
# =====================================================================
class HomographyFilter:
    """第一重过滤：场地单应性矩阵 H 的滑动窗口平滑器"""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.h_history = []

    def update(self, new_H):
        if new_H is None:
            return None
        self.h_history.append(new_H)
        if len(self.h_history) > self.window_size:
            self.h_history.pop(0)

        smoothed_H = np.mean(self.h_history, axis=0)
        smoothed_H = smoothed_H / smoothed_H[2, 2]  # 重新归一化
        return smoothed_H


class RadarDrawer:
    """第二重过滤：2D 红点坐标的平滑器与渲染器"""

    def __init__(self, window_size=6):
        self.history = []
        self.window_size = window_size
        self.trail = []

        # 雷达UI设置 (放大 12 倍画图)
        self.map_scale = 12
        self.map_w = int(10.97 * self.map_scale * 2)
        self.map_h = int(23.77 * self.map_scale * 1.5)
        self.center_x = self.map_w // 2
        self.center_y = self.map_h // 2

    def smooth_point(self, real_coord):
        """轨迹点的滑动平均"""
        self.history.append(real_coord)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return np.mean(self.history, axis=0)

    def draw_minimap(self, frame, current_real_coord):
        """渲染 UI 层"""
        overlay = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
        # 画绿色的球场底板
        cv2.rectangle(overlay, (0, 0), (self.map_w, self.map_h), (80, 120, 80), -1)

        # 简单绘制外框和球网
        scale_x, scale_y = int(5.485 * self.map_scale), int(11.885 * self.map_scale)
        top_left = (self.center_x - scale_x, self.center_y - scale_y)
        bottom_right = (self.center_x + scale_x, self.center_y + scale_y)
        cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), 2)
        cv2.line(overlay, (top_left[0], self.center_y), (bottom_right[0], self.center_y), (255, 255, 255), 2)

        # 尾迹消散逻辑
        if current_real_coord is not None:
            self.trail.append({'x': current_real_coord[0], 'y': current_real_coord[1], 'alpha': 1.0})

        for p in self.trail: p['alpha'] -= 0.03
        self.trail = [p for p in self.trail if p['alpha'] > 0]

        # 渲染轨迹
        for p in self.trail:
            draw_x = int(p['x'] * self.map_scale) + self.center_x
            draw_y = int(-p['y'] * self.map_scale) + self.center_y  # Y轴反转，符合俯视直觉
            color = (0, int(255 * p['alpha']), int(255 * p['alpha']))  # BGR 黄色尾迹
            cv2.circle(overlay, (draw_x, draw_y), 3, color, -1)

        # 绘制最新位置的大红点
        if self.trail:
            latest = self.trail[-1]
            draw_x = int(latest['x'] * self.map_scale) + self.center_x
            draw_y = int(-latest['y'] * self.map_scale) + self.center_y
            cv2.circle(overlay, (draw_x, draw_y), 6, (0, 0, 255), -1)

        # 贴图混合
        alpha = 0.7
        roi = frame[20:20 + self.map_h, 20:20 + self.map_w]
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
        return frame


# =====================================================================
# 4. 主控引擎
# =====================================================================
def main():
    print("⏳ 正在加载 YOLO 视觉大模型...")
    court_model = YOLO(COURT_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 初始化双重滤波器
    radar = RadarDrawer(window_size=6)
    h_filter = HomographyFilter(window_size=5)

    prev_smoothed_H = None

    print("开始生成轨迹...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # ---------------------------------------------------------
        # [A] 场地线识别与动态矩阵更新
        # ---------------------------------------------------------
        court_res = court_model.predict(frame, conf=0.3, half=True, verbose=False)[0]

        valid_pixel_pts, valid_phys_pts, valid_weights = [], [], []
        corner_count = 0

        if court_res.keypoints is not None and len(court_res.keypoints.data) > 0:
            kpts = court_res.keypoints.data[0].cpu().numpy()
            for i, pt in enumerate(kpts):
                x, y, conf = pt
                if conf > 0.4:
                    valid_pixel_pts.append([x, y])
                    valid_phys_pts.append(COURT_14_PTS_PHYSICAL[i])
                    valid_weights.append(BASE_WEIGHTS[i] * conf)
                    if i < 4: corner_count += 1

            # 只有当有效点足够时，才计算新的 H 矩阵
            if corner_count >= 2 and len(valid_pixel_pts) >= 4:
                raw_H = get_weighted_homography(
                    np.array(valid_phys_pts, dtype=np.float32),
                    np.array(valid_pixel_pts, dtype=np.float32),
                    np.array(valid_weights, dtype=np.float32)
                )

                # 核心过滤：将原始矩阵传入滤波器，获取平滑矩阵
                if raw_H is not None:
                    prev_smoothed_H = h_filter.update(raw_H)

        # ---------------------------------------------------------
        # [B] 人体姿态与映射
        # ---------------------------------------------------------
        real_coord_smoothed = None
        if prev_smoothed_H is not None:
            pose_res = pose_model.predict(frame, verbose=False)[0]

            if pose_res.keypoints is not None and len(pose_res.keypoints.data) > 0:
                poses = pose_res.keypoints.data.cpu().numpy()

                # 【占位符】：这里目前用 Y 坐标最大的启发式规则锁定球员
                # 等你研究“指定运动员”逻辑时，重点修改这里的 best_person 筛选条件即可
                best_person = None
                max_y = -1

                for pose in poses:
                    l_foot, r_foot = pose[15], pose[16]
                    if l_foot[2] > 0.3 and r_foot[2] > 0.3:
                        avg_y = (l_foot[1] + r_foot[1]) / 2
                        if avg_y > max_y:
                            max_y = avg_y
                            best_person = pose

                if best_person is not None:
                    feet_center_px = (best_person[15][:2] + best_person[16][:2]) / 2.0
                    pt = np.array([[[feet_center_px[0], feet_center_px[1]]]], dtype=np.float32)

                    # 使用平滑后的矩阵进行坐标转换
                    real_coord = cv2.perspectiveTransform(pt, prev_smoothed_H)[0][0]

                    # 坐标点滑动平滑
                    real_coord_smoothed = radar.smooth_point(real_coord)

        # ---------------------------------------------------------
        # [C] 雷达渲染
        # ---------------------------------------------------------
        frame = radar.draw_minimap(frame, real_coord_smoothed)
        cv2.imshow("Tennis Pro Radar (Ultimate)", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("终极渲染完毕！视频已保存。")


if __name__ == "__main__":
    main()