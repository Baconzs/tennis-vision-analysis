"""
hard_negative_reviewer.py — 交互式帧审核 + 标注工具（合二为一）

操作说明：
    左键拖拽        : 画检测框
    0               : 将最后一个框标为 player_near (class 0)
    1               : 将最后一个框标为 player_far  (class 1)
    Z               : 撤销最后一个框
    K / Space       : 保存标注并跳到下一张
    D               : 删除此帧（不合格），标记片段，跳下一张
    A / ←           : 回到上一张
    Q / ESC         : 保存进度并退出

输出：
    data/person_sorter/hard_negatives/labels/  — YOLO 格式标注 txt
    logs/low_quality_clips.txt                 — 不合格片段列表
    data/person_sorter/hard_negatives/reviewed.txt — 断点续跑
"""

import os
import csv
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

IMAGES_DIR   = os.path.join(_PROJECT_DIR, "data", "person_sorter", "hard_negatives", "images")
LABELS_DIR   = os.path.join(_PROJECT_DIR, "data", "person_sorter", "hard_negatives", "labels")
MANIFEST_PATH = os.path.join(_PROJECT_DIR, "data", "person_sorter", "hard_negatives", "manifest.csv")
REVIEWED_PATH = os.path.join(_PROJECT_DIR, "data", "person_sorter", "hard_negatives", "reviewed.txt")
LOW_QUALITY_PATH = os.path.join(_PROJECT_DIR, "logs", "low_quality_clips.txt")

CLASS_COLORS = {0: (0, 200, 255), 1: (200, 100, 255)}  # near=黄, far=紫
CLASS_NAMES  = {0: "player_near", 1: "player_far"}


def _imread(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_manifest():
    mapping = {}
    if not os.path.exists(MANIFEST_PATH):
        return mapping
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["filename"]] = row["source_video"]
    return mapping


def load_reviewed():
    if not os.path.exists(REVIEWED_PATH):
        return set()
    with open(REVIEWED_PATH, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_reviewed(reviewed):
    with open(REVIEWED_PATH, "w", encoding="utf-8") as f:
        for name in sorted(reviewed):
            f.write(name + "\n")


def load_low_quality():
    if not os.path.exists(LOW_QUALITY_PATH):
        return set()
    with open(LOW_QUALITY_PATH, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_low_quality(clips):
    os.makedirs(os.path.dirname(LOW_QUALITY_PATH), exist_ok=True)
    with open(LOW_QUALITY_PATH, "w", encoding="utf-8") as f:
        for clip in sorted(clips):
            f.write(clip + "\n")


def save_labels(img_path: Path, boxes: list):
    """boxes: list of (cls, x1, y1, x2, y2) in pixel coords"""
    os.makedirs(LABELS_DIR, exist_ok=True)
    label_path = os.path.join(LABELS_DIR, img_path.stem + ".txt")
    frame = _imread(img_path)
    if frame is None:
        return
    h, w = frame.shape[:2]
    with open(label_path, "w") as f:
        for cls, x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def render(base_frame, boxes, drawing, pt1, pt2, idx, total, filename, conf):
    """重绘当前帧：已有框 + 正在画的框 + UI 提示"""
    frame = base_frame.copy()
    h, w = frame.shape[:2]

    # 已确认的框
    for cls, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES.get(cls, str(cls))
        cv2.putText(frame, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 正在拖拽的框（白色虚线效果用白色代替）
    if drawing and pt1 and pt2:
        cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 1)

    # 底部信息栏
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"[{idx+1}/{total}]  {filename}", (10, h - 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if conf:
        cv2.putText(frame, f"min_conf: {conf}   boxes: {len(boxes)}", (10, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 100), 1)
    cv2.putText(frame, "Draw box: auto near->far  |  Z: undo", (10, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 50), 1)
    cv2.putText(frame, "K/Space: save & next  |  D: bad frame  |  A: prev  |  Q: quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 50), 1)
    return frame


def main():
    if not os.path.exists(IMAGES_DIR):
        print(f"图像目录不存在: {IMAGES_DIR}")
        print("请先运行 hard_negative_extractor.py 提取帧")
        return

    all_images = sorted(
        Path(IMAGES_DIR) / name
        for name in os.listdir(IMAGES_DIR)
        if name.lower().endswith(".jpg")
    )
    if not all_images:
        print("没有找到待审核的帧图像")
        return

    manifest    = load_manifest()
    reviewed    = load_reviewed()
    low_quality = load_low_quality()

    pending = [p for p in all_images if p.name not in reviewed]
    if not pending:
        print("所有帧已审核完毕！")
        print(f"不合格片段共 {len(low_quality)} 个，见: {LOW_QUALITY_PATH}")
        return

    print(f"待审核: {len(pending)} 帧 (已跳过 {len(reviewed)} 帧)")

    conf_map = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                conf_map[row["filename"]] = row.get("min_conf", "")

    win = "Annotator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    # 鼠标状态
    mouse = {"drawing": False, "pt1": None, "pt2": None, "boxes": []}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["drawing"] = True
            mouse["pt1"] = (x, y)
            mouse["pt2"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and mouse["drawing"]:
            mouse["pt2"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["drawing"] = False
            mouse["pt2"] = (x, y)
            x1, y1 = mouse["pt1"]
            x2, y2 = mouse["pt2"]
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                # 自动分配类别：没有0就画0，有0没有1就画1
                existing = {b[0] for b in mouse["boxes"]}
                if 0 not in existing:
                    cls = 0
                elif 1 not in existing:
                    cls = 1
                else:
                    cls = 0  # 超过两个框时循环从0开始
                mouse["boxes"].append((cls, min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)))

    cv2.setMouseCallback(win, on_mouse)

    history = []  # list of (img_path, action)
    idx = 0

    while idx < len(pending):
        img_path = pending[idx]
        base = _imread(img_path)
        if base is None:
            reviewed.add(img_path.name)
            idx += 1
            continue

        conf_str = conf_map.get(img_path.name, "")
        mouse["boxes"] = []

        while True:
            frame = render(base, mouse["boxes"], mouse["drawing"],
                           mouse["pt1"], mouse["pt2"],
                           idx, len(pending), img_path.name, conf_str)

            cv2.imshow(win, frame)
            key = cv2.waitKey(30) & 0xFF

            if key == 255:  # 无按键
                continue

            # 撤销最后一个框
            if key in (ord('z'), ord('Z')):
                if mouse["boxes"]:
                    mouse["boxes"].pop()

            # 保存标注，跳下一张
            elif key in (ord('k'), ord('K'), 32):
                if mouse["boxes"]:
                    save_labels(img_path, mouse["boxes"])
                reviewed.add(img_path.name)
                history.append((img_path, "keep"))
                idx += 1
                break

            # 删除帧，标记不合格
            elif key in (ord('d'), ord('D')):
                try:
                    os.remove(str(img_path))
                except OSError:
                    pass
                reviewed.add(img_path.name)
                source = manifest.get(img_path.name, "")
                if source:
                    low_quality.add(source)
                history.append((img_path, "delete"))
                idx += 1
                break

            # 回到上一张
            elif key in (ord('a'), ord('A'), 81, 2):
                if history:
                    last_path, last_action = history.pop()
                    reviewed.discard(last_path.name)
                    if last_action == "delete":
                        source = manifest.get(last_path.name, "")
                        if source:
                            low_quality.discard(source)
                    idx = max(0, idx - 1)
                break

            # 退出
            elif key in (ord('q'), ord('Q'), 27):
                cv2.destroyAllWindows()
                save_reviewed(reviewed)
                save_low_quality(low_quality)
                remaining = len([p for p in all_images if p.name not in reviewed])
                print(f"\n进度已保存，剩余 {remaining} 帧未审核")
                print(f"不合格片段: {len(low_quality)} 个 -> {LOW_QUALITY_PATH}")
                return

    cv2.destroyAllWindows()
    save_reviewed(reviewed)
    save_low_quality(low_quality)
    print(f"\n全部审核完毕！不合格片段: {len(low_quality)} 个 -> {LOW_QUALITY_PATH}")
    print(f"标注文件保存在: {LABELS_DIR}")


if __name__ == "__main__":
    main()
