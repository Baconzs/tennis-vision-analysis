"""debug_vision.py — 可视化调试工具

功能：将追踪/检测结果叠加到视频帧上，用于调试和验证流水线输出
"""
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.optimize import least_squares
from collections import defaultdict, deque

# =====================================================================
# 1. 路径与模型配置
# =====================================================================
VIDEO_PATH = "data/rallies_annotated/rally_001_19.8s/raw_clip.mp4"
OUTPUT_PATH = r"output_god_mode.mp4"

COURT_MODEL_PATH = "runs/court_finetune/court_14pts_ultimate/weights/best.pt"
TRACKER_MODEL_PATH = r"best.pt"  # 你的专属 YOLO26x 运动员追踪模型
POSE_MODEL_PATH = r"yolo11x-pose.pt"

# 球场 14 个关键点的物理坐标
COURT_PHYSICAL = np.array([
    [-5.485, -11.885], [5.485, -11.885], [5.485, 11.885], [-5.485, 11.885],
    [0.000, -11.885], [0.000, 11.885], [-4.115, -6.400], [4.115, -6.400],
    [0.000, -6.400], [-4.115, 6.400], [4.115, 6.400], [0.000, 6.400],
    [-5.485, 0.000], [5.485, 0.000]
], dtype=np.float32)

COURT_LINES = [(0, 1), (2, 3), (0, 3), (1, 2), (6, 7), (9, 10), (4, 5), (12, 13)]


# =====================================================================
# 2. 数学映射与时序清洗 (已彻底修复所有越界与 Bug)
# =====================================================================
def get_weighted_homography(phys_pts, pixel_pts, weights):
    """带权重的单应性变换，利用 LM 算法优化投影残差"""
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


def interpolate_track(track_dict, max_gap=20):
    """利用线性插值补全追踪断层，并使用 EMA 平滑坐标，安全继承额外数据"""
    if not track_dict: return {}
    f_indices = sorted(track_dict.keys())

    # 提取核心用于计算的数值数据
    data = []
    for f in f_indices:
        d = track_dict[f]
        data.append([f, d['real'][0], d['real'][1], d['box'][0], d['box'][1], d['box'][2], d['box'][3]])

    df = pd.DataFrame(data, columns=['f', 'x', 'y', 'x1', 'y1', 'x2', 'y2'])
    df = df.set_index('f')

    # 重新索引，产生 NaN 用于插值 (修复拼写错误)
    full_idx = np.arange(f_indices[0], f_indices[-1] + 1)
    df = df.reindex(full_idx)

    # 动态限制 max_gap，防止 pandas 滑动窗口越界崩溃
    actual_limit = min(max_gap, len(df) - 1)
    if actual_limit > 0:
        df = df.interpolate(method='linear', limit=actual_limit)

    # EMA 指数加权移动平均平滑
    df = df.ewm(alpha=0.3).mean().dropna()

    new_track = {}
    for f, row in df.iterrows():
        f_int = int(f)
        # 妥善继承骨骼点信息
        original_data = track_dict.get(f_int, {})
        original_kpts = original_data.get('keypoints', None)

        new_track[f_int] = {
            "real": np.array([row['x'], row['y']]),
            "box": [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])],
            "keypoints": original_kpts
        }
    return new_track


# =====================================================================
# 3. 渐变拖尾雷达图
# =====================================================================
class RadarRenderer:
    def __init__(self, w=300, h=600):
        self.w, self.h = w, h
        self.scale = 22
        self.cx, self.cy = w // 2, h // 2
        # deque 实现最多保留 25 帧历史用于拖尾
        self.history = defaultdict(lambda: deque(maxlen=25))

    def draw(self, frame, active_players):
        radar = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # 球场底图与白线
        cv2.rectangle(radar, (0, 0), (self.w, self.h), (25, 45, 25), -1)
        hw, hh = 5.485 * self.scale, 11.885 * self.scale
        cv2.rectangle(radar, (int(self.cx - hw), int(self.cy - hh)), (int(self.cx + hw), int(self.cy + hh)),
                      (230, 230, 230), 2)
        cv2.line(radar, (int(self.cx - hw), self.cy), (int(self.cx + hw), self.cy), (230, 230, 230), 1)

        for tid, data in active_players.items():
            pos = data["real"]
            px, py = int(pos[0] * self.scale + self.cx), int(-pos[1] * self.scale + self.cy)
            self.history[tid].append((px, py))

            # 绘制从暗到亮的渐变拖尾
            h_list = list(self.history[tid])
            for i in range(1, len(h_list)):
                alpha = i / len(h_list)
                color = (0, int(200 * alpha), int(255 * alpha))
                cv2.line(radar, h_list[i - 1], h_list[i], color, max(1, int(4 * alpha)))

            # 绘制当前头节点
            cv2.circle(radar, (px, py), 6, (0, 255, 255), -1)

        # Alpha 融合到原图
        roi = frame[20:20 + self.h, 20:20 + self.w]
        cv2.addWeighted(radar, 0.8, roi, 0.2, 0, roi)
        return frame


# =====================================================================
# 4. 主系统 (Pipeline)
# =====================================================================
def main():
    print("⏳ 正在加载模型权重...")
    court_model = YOLO(COURT_MODEL_PATH)
    tracker_model = YOLO(TRACKER_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps, width, height = int(cap.get(5)), int(cap.get(3)), int(cap.get(4))

    tracks_db = defaultdict(dict)
    h_db = {}
    f_idx = 0
    last_H = None

    print("\n[Phase 1/3] 正在提取视觉特征 (Top-Down 开启)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- 1. 球场检测 (修复 IndexError) ---
        c_res = court_model.predict(frame, conf=0.3, verbose=False)[0]
        if c_res.keypoints is not None and len(c_res.keypoints.data) > 0:
            kpts = c_res.keypoints.data[0].cpu().numpy()
            if len(kpts) >= 4:
                mask = kpts[:, 2] > 0.4
                if np.sum(mask) >= 4:
                    last_H = get_weighted_homography(COURT_PHYSICAL[mask], kpts[mask, :2], kpts[mask, 2])
        h_db[f_idx] = last_H

        # --- 2. 运动员追踪与降级寻脚 ---
        if last_H is not None:
            t_res = tracker_model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
            if t_res.boxes is not None and t_res.boxes.id is not None:
                ids = t_res.boxes.id.int().cpu().tolist()
                bboxes = t_res.boxes.xyxy.cpu().numpy()
                H_inv = np.linalg.inv(last_H)

                for i, tid in enumerate(ids):
                    bx = bboxes[i].astype(int)

                    # 扩大 Bbox 50%，增加姿态模型的感受野
                    bw, bh = bx[2] - bx[0], bx[3] - bx[1]
                    pad_x = int(bw * 0.25) + 10
                    pad_y = int(bh * 0.25) + 10

                    cx1, cy1 = max(0, bx[0] - pad_x), max(0, bx[1] - pad_y)
                    cx2, cy2 = min(width, bx[2] + pad_x), min(height, bx[3] + pad_y)
                    athlete_crop = frame[cy1:cy2, cx1:cx2]

                    feet_px = np.array([(bx[0] + bx[2]) / 2, bx[3]])  # 兜底坐标：框底中心
                    kpts_global = None

                    # 只要截图有效，就送入 Pose 模型提取细节
                    if athlete_crop.shape[0] >= 10 and athlete_crop.shape[1] >= 10:
                        p_res = pose_model.predict(athlete_crop, imgsz=192, verbose=False)[0]
                        if p_res.keypoints is not None and len(p_res.keypoints.data) > 0:
                            # 修复双重偏移 Bug：复制数组，并一次性还原到全图坐标系
                            kp = p_res.keypoints.data[0].cpu().numpy().copy()
                            kp[:, 0] += cx1
                            kp[:, 1] += cy1
                            kpts_global = kp

                            # 层级化降级寻脚算法
                            l_ankle, r_ankle = kp[15], kp[16]
                            l_knee, r_knee = kp[13], kp[14]

                            if l_ankle[2] > 0.35 and r_ankle[2] > 0.35:  # 双脚踝
                                feet_px = (l_ankle[:2] + r_ankle[:2]) / 2.0
                            elif l_ankle[2] > 0.35:  # 单左脚
                                feet_px = l_ankle[:2]
                            elif r_ankle[2] > 0.35:  # 单右脚
                                feet_px = r_ankle[:2]
                            elif l_knee[2] > 0.4 and r_knee[2] > 0.4:  # 膝盖兜底(网带遮挡)
                                feet_px = (l_knee[:2] + r_knee[:2]) / 2.0

                    # 坐标映射到物理 2D 平面
                    pt = np.array([[[feet_px[0], feet_px[1]]]], dtype=np.float32)
                    real_pos = cv2.perspectiveTransform(pt, H_inv)[0][0]
                    tracks_db[tid][f_idx] = {"real": real_pos, "box": bx, "keypoints": kpts_global}

        f_idx += 1
        if f_idx % 100 == 0: print(f"   已提取 {f_idx} 帧...")

    print("\n[Phase 2/3] 正在进行时序插值补偿与数据清洗...")
    final_tracks = {}
    for tid in tracks_db.keys():
        processed = interpolate_track(tracks_db[tid])
        if len(processed) > 10: final_tracks[tid] = processed

    # 筛选存活时间最长（积分最高）的两人作为 P1 和 P2
    id_scores = {tid: len(data) for tid, data in final_tracks.items()}
    top_2_ids = sorted(id_scores, key=id_scores.get, reverse=True)[:2]

    print(f"\n[Phase 3/3] 正在渲染并写入视频 (锁定 ID: {top_2_ids})...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    radar = RadarRenderer()

    for f in range(f_idx):
        ret, frame = cap.read()
        if not ret: break

        # 绘制球场线
        if h_db[f] is not None:
            H = h_db[f]
            for l in COURT_LINES:
                p1 = np.append(COURT_PHYSICAL[l[0]], 1.0)
                p2 = np.append(COURT_PHYSICAL[l[1]], 1.0)
                px1 = (H @ p1);
                px1 /= px1[2]
                px2 = (H @ p2);
                px2 /= px2[2]
                cv2.line(frame, (int(px1[0]), int(px1[1])), (int(px2[0]), int(px2[1])), (0, 255, 0), 2)

        # 绘制运动员与姿态
        active_this_frame = {}
        for tid in top_2_ids:
            if f in final_tracks[tid]:
                d = final_tracks[tid][f]
                active_this_frame[tid] = d

                # 画追踪框
                b = d["box"]
                if b is not None:
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
                    cv2.putText(frame, f"ID {tid}", (b[0], b[1] - 10), 0, 0.6, (0, 255, 255), 2)

                # 画红点骨骼
                kpts = d.get("keypoints")
                if kpts is not None:
                    for kp in kpts:
                        if kp[2] > 0.4:
                            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

        # 叠加雷达
        frame = radar.draw(frame, active_this_frame)

        writer.write(frame)
        cv2.imshow("God Mode Ultimate Tracker", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n毕业设计核心模块大功告成！视频已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()