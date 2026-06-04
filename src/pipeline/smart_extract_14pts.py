"""smart_extract_14pts.py — 智能采样标注工具

功能：从比赛视频中智能采样帧，用球场模型预标注14个关键点，生成训练数据
"""
import os
import cv2
import random
import numpy as np
from ultralytics import YOLO
from scipy.optimize import least_squares

# =====================================================================
# 1. 物理坐标与权重定义
# =====================================================================
COURT_14_PTS_PHYSICAL = np.array([
    [-5.485, -11.885], [5.485, -11.885], [5.485, 11.885], [-5.485, 11.885],
    [0.000, -11.885], [0.000, 11.885],
    [-4.115, -6.400], [4.115, -6.400], [0.000, -6.400],
    [-4.115, 6.400], [4.115, 6.400], [0.000, 6.400],
    [-5.485, 0.000], [5.485, 0.000]
], dtype=np.float32)

BASE_WEIGHTS = np.array([9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)


# =====================================================================
# 2. SciPy 加权矩阵求解器
# =====================================================================
def reprojection_residuals(h_elements, src_pts, dst_pts, weights):
    H = np.append(h_elements, 1.0).reshape(3, 3)
    src_pts_3d = np.concatenate([src_pts, np.ones((len(src_pts), 1))], axis=1)
    proj_pts_3d = (H @ src_pts_3d.T).T
    proj_pts_3d[:, 2] = np.where(proj_pts_3d[:, 2] == 0, 1e-7, proj_pts_3d[:, 2])
    proj_pts_2d = proj_pts_3d[:, :2] / proj_pts_3d[:, 2:]

    # 分离 X 和 Y 的误差，生成 2N 个残差，满足 lm 算法要求
    errors = proj_pts_2d - dst_pts
    weighted_errors = errors * weights[:, np.newaxis]
    return weighted_errors.flatten()


def get_weighted_homography(phys_pts, pixel_pts, weights):
    H_init, _ = cv2.findHomography(phys_pts, pixel_pts, cv2.RANSAC, 5.0)
    if H_init is None: return None
    h_initial_guess = (H_init / H_init[2, 2]).flatten()[:8]
    res = least_squares(
        reprojection_residuals,
        x0=h_initial_guess,
        args=(phys_pts, pixel_pts, weights),
        method='lm'
    )
    return np.append(res.x, 1.0).reshape(3, 3)


# =====================================================================
# 3. 智能抽帧主控逻辑
# =====================================================================
def smart_sampling(video_folder, model_path, output_dir, samples_per_video=20):
    model = YOLO(model_path)
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    for v_file in video_files:
        v_path = os.path.join(video_folder, v_file)
        cap = cv2.VideoCapture(v_path)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_f < 60: continue

        print(f"正在扫描视频: {v_file}")
        found_in_video = 0
        attempts = 0

        while found_in_video < samples_per_video and attempts < samples_per_video * 5:
            attempts += 1
            idx = random.randint(0, total_f - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            # 半精度极速推理
            results = model.predict(frame, conf=0.3, verbose=False, half=True)[0]

            if results.boxes is not None and len(results.boxes) > 0:
                box = results.boxes.xywhn[0].cpu().numpy()
                area = box[2] * box[3]

                # 门槛1：场地不能太小
                if area > 0.15:
                    kpts = results.keypoints.data[0].cpu().numpy()

                    valid_pixel_pts = []
                    valid_phys_pts = []
                    valid_weights = []
                    corner_count = 0

                    # 提取高置信度点并叠加权重
                    for i, pt in enumerate(kpts):
                        x, y, conf = pt
                        if conf > 0.4:
                            valid_pixel_pts.append([x, y])
                            valid_phys_pts.append(COURT_14_PTS_PHYSICAL[i])
                            valid_weights.append(BASE_WEIGHTS[i] * conf)
                            if i < 4:
                                corner_count += 1

                    # 门槛2：核心角点 >= 2 且总有效点数 >= 4
                    if corner_count >= 2 and len(valid_pixel_pts) >= 4:

                        # 核心计算：求解出最优的加权单应性矩阵
                        H = get_weighted_homography(
                            np.array(valid_phys_pts, dtype=np.float32),
                            np.array(valid_pixel_pts, dtype=np.float32),
                            np.array(valid_weights, dtype=np.float32)
                        )

                        if H is not None:
                            # 降维打击：用算出的完美矩阵，重新生成14个点的理想像素坐标
                            # 这样保存下来的标签天然就是刚性完美的，微调时极其舒服
                            phys_3d = np.concatenate([COURT_14_PTS_PHYSICAL, np.ones((14, 1))], axis=1)
                            proj_3d = (H @ phys_3d.T).T
                            perfect_kpts = proj_3d[:, :2] / proj_3d[:, 2:]

                            # 保存图片
                            save_name = f"{os.path.splitext(v_file)[0]}_f{idx}"
                            img_path = os.path.join(img_dir, f"{save_name}.jpg")
                            cv2.imencode('.jpg', frame)[1].tofile(img_path)

                            # 保存标签
                            h, w = frame.shape[:2]
                            label_str = f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"

                            for pkp in perfect_kpts:
                                px, py = pkp[0], pkp[1]
                                # 智能判定可见度：如果通过矩阵算出来的点在画面外，设为 0
                                vis = 2 if (0 <= px <= w and 0 <= py <= h) else 0
                                label_str += f" {px / w:.6f} {py / h:.6f} {vis}"

                            with open(os.path.join(lbl_dir, f"{save_name}.txt"), 'w') as f:
                                f.write(label_str + "\n")

                            found_in_video += 1
                            print(f"  完美帧 ({found_in_video}/{samples_per_video}) | 角点:{corner_count}/4")

    cap.release()
    print(f"预标注完成！共存放在 {output_dir}")


if __name__ == "__main__":
    import os as _os
    _PROJECT_DIR = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    VIDEO_PATH = _os.path.join(_PROJECT_DIR, "videos")
    MODEL_PATH = _os.path.join(_PROJECT_DIR, "runs", "court_finetune", "court_14pts_weighted", "weights", "best.pt")
    OUTPUT_DIR = _os.path.join(_PROJECT_DIR, "data", "court_finetune")
    smart_sampling(VIDEO_PATH, MODEL_PATH, OUTPUT_DIR, samples_per_video=20)