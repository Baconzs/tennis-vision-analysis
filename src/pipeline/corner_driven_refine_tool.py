"""corner_driven_refine_tool.py — 球场角点驱动的标注精修工具

功能：对已有标注进行角点对齐优化，提升球场关键点标注精度
"""
import cv2
import os
import glob
import numpy as np

# 14点标准物理坐标
PHYS_14 = np.array([
    [-5.485, -11.885], [5.485, -11.885], [5.485, 11.885], [-5.485, 11.885],
    [0.000, -11.885], [0.000, 11.885],
    [-4.115, -6.400], [4.115, -6.400], [0.000, -6.400],
    [-4.115, 6.400], [4.115, 6.400], [0.000, 6.400],
    [-5.485, 0.000], [5.485, 0.000]
], dtype=np.float32)

COURT_LINE_INDICES = [
    (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (6, 7), (9, 10), (8, 11), (12, 13)
]


class UltimateRefiner:
    def __init__(self, img_dir, lbl_dir, margin=200):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.margin = margin

        # 注释掉自动删除，因为我们需要允许有没有标签的“新图片”存在
        # self.clean_orphaned_images()

        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.idx = 0
        self.pts = []
        self.selected_idx = -1
        self.img = None
        self.raw_h, self.raw_w = 0, 0
        self.window = "Tennis Label Refiner - Ultimate"

    def init_default_template(self):
        """核心功能 1：如果没有任何点，在画面中央初始化一个标准的场地模板"""
        self.pts = []
        # 将物理坐标等比例缩放并移到画面中央
        scale = (self.raw_w * 0.5) / 11.0  # 让场地宽度占屏幕 50%
        offset_x = self.raw_w / 2
        offset_y = self.raw_h / 2

        for p in PHYS_14:
            px = p[0] * scale + offset_x
            py = p[1] * scale + offset_y
            self.pts.append([px, py, 2.0])  # 默认设为可见

    def update_others(self):
        if len(self.pts) < 14: return
        src_4 = PHYS_14[:4]
        dst_4 = np.array([[p[0], p[1]] for p in self.pts[:4]], dtype=np.float32)
        H, _ = cv2.findHomography(src_4, dst_4)
        if H is not None:
            others_phys = PHYS_14[4:].reshape(-1, 1, 2)
            others_dst = cv2.perspectiveTransform(others_phys, H)
            for i in range(10):
                self.pts[i + 4][0], self.pts[i + 4][1] = others_dst[i][0][0], others_dst[i][0][1]

    def load(self):
        if self.idx >= len(self.img_files): return False
        path = self.img_files[self.idx]
        self.img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        self.raw_h, self.raw_w = self.img.shape[:2]
        lbl = os.path.join(self.lbl_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")

        self.pts = []
        loaded = False
        if os.path.exists(lbl):
            with open(lbl, 'r') as f:
                d = f.read().strip().split()
                if len(d) >= 47:  # 包含完整的 14 个点
                    k = [float(x) for x in d[5:]]
                    for i in range(0, len(k), 3):
                        self.pts.append([k[i] * self.raw_w, k[i + 1] * self.raw_h, k[i + 2]])
                    loaded = True
                elif len(d) == 0:
                    # 这是一个被标记为空的负样本
                    loaded = True

                    # 如果文件不存在，说明模型漏检了，立刻给一个默认模板让你拖拽！
        if not loaded:
            self.init_default_template()
            print(f"模型未检测到场地，已生成默认模板: {os.path.basename(path)}")

        return True

    def save(self, is_empty=False):
        """核心功能 2：支持保存空负样本"""
        path = self.img_files[self.idx]
        lbl = os.path.join(self.lbl_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")

        if is_empty or len(self.pts) == 0:
            # 保存为一个空 txt 文件，这会告诉 YOLO "这张图里什么都没有"
            with open(lbl, 'w') as f: f.write("")
            print(f"已保存为 [空/负样本]: {os.path.basename(lbl)}")
            return

        # 正常的边界框与点位保存
        xs, ys = [p[0] for p in self.pts], [p[1] for p in self.pts]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        bw, bh = max((xmax - xmin) / self.raw_w, 0.01), max((ymax - ymin) / self.raw_h, 0.01)
        bx, by = (xmin + xmax) / 2 / self.raw_w, (ymin + ymax) / 2 / self.raw_h

        line = f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}"
        for p in self.pts:
            vis = 2 if (0 <= p[0] <= self.raw_w and 0 <= p[1] <= self.raw_h) else 0
            line += f" {p[0] / self.raw_w:.6f} {p[1] / self.raw_h:.6f} {vis}"
        with open(lbl, 'w') as f:
            f.write(line + "\n")
        print(f"已保存标准标签: {os.path.basename(lbl)}")

    def delete_current(self):
        img_path = self.img_files[self.idx]
        lbl_path = os.path.join(self.lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(lbl_path): os.remove(lbl_path)
        print(f"已彻底删除文件: {os.path.basename(img_path)}")
        self.img_files.pop(self.idx)

    def mouse(self, event, x, y, flags, param):
        if len(self.pts) == 0: return  # 空样本不支持拖动

        rx, ry = x - self.margin, y - self.margin
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(4):
                if np.hypot(rx - self.pts[i][0], ry - self.pts[i][1]) < 20:
                    self.selected_idx = i
                    break
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_idx = -1
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_idx != -1:
            self.pts[self.selected_idx][0], self.pts[self.selected_idx][1] = rx, ry
            self.update_others()

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse)

        while self.idx < len(self.img_files):
            if not self.load(): break
            while True:
                canvas = cv2.copyMakeBorder(self.img, self.margin, self.margin, self.margin, self.margin,
                                            cv2.BORDER_CONSTANT, value=[30, 30, 30])
                cv2.rectangle(canvas, (self.margin, self.margin),
                              (self.margin + self.raw_w, self.margin + self.raw_h), (100, 100, 100), 2)

                # 只有在有点位时才绘制连线和点
                if len(self.pts) > 0:
                    for start_idx, end_idx in COURT_LINE_INDICES:
                        p1 = (int(self.pts[start_idx][0] + self.margin), int(self.pts[start_idx][1] + self.margin))
                        p2 = (int(self.pts[end_idx][0] + self.margin), int(self.pts[end_idx][1] + self.margin))
                        cv2.line(canvas, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

                    for i, p in enumerate(self.pts):
                        dx, dy = int(p[0] + self.margin), int(p[1] + self.margin)
                        color = (0, 0, 255) if i < 4 else (255, 200, 0)
                        cv2.circle(canvas, (dx, dy), 6, color, -1)
                else:
                    # 负样本提示
                    cv2.putText(canvas, "NO COURT (Negative Sample)", (self.margin + 50, self.margin + 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                menu = [
                    f"FILE: {self.idx + 1}/{len(self.img_files)}",
                    "--------------------------",
                    "[SPACE] - Save & Next",
                    "[ E ]   - Save as EMPTY (Negative)",
                    "[ R ]   - Reset to Default Temp",
                    "[ D ]   - Skip to Next",
                    "[ A ]   - Back to Prev",
                    "[ X ]   - DELETE Data",
                    "[ Q ]   - Quit Tool",
                    "--------------------------"
                ]
                for i, text in enumerate(menu):
                    cv2.putText(canvas, text, (20, 40 + i * 30), 1, 1.2, (0, 255, 255), 2)

                cv2.imshow(self.window, canvas)
                key = cv2.waitKey(10) & 0xFF

                if key == ord(' '):  # 正常保存
                    self.save(is_empty=False);
                    self.idx += 1;
                    break
                elif key == ord('e'):  # 保存为负样本
                    self.save(is_empty=True);
                    self.idx += 1;
                    break
                elif key == ord('r'):  # 重新生成默认模板
                    self.init_default_template()
                elif key == ord('d'):
                    self.idx += 1;
                    break
                elif key == ord('a'):
                    self.idx = max(0, self.idx - 1);
                    break
                elif key == ord('x'):
                    self.delete_current();
                    break
                elif key == ord('q'):
                    return

        print("所有图片标注/筛选完成！")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    IMG_DIR = r"./Second_Train_Dataset/images"
    LBL_DIR = r"./Second_Train_Dataset/labels"
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)

    refiner = UltimateRefiner(IMG_DIR, LBL_DIR, margin=200)
    refiner.run()