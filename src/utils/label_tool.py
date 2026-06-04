"""label_tool.py — 球员边界框手动标注工具（YOLO 格式）

功能：在图片上拖拽绘制 bounding box，标注近端/远端球员，保存为 YOLO txt 格式
"""
import cv2
import os
import glob
import numpy as np

# ================= 配置区域 =================
# 假设脚本和 image 文件夹在同一个目录下
IMAGE_DIR = "data/image"  # 根据你的报错信息，我把路径更新为 data/image
LABEL_DIR = "data/labels"  # 对应生成 labels 文件夹

# YOLO 类别定义
CLASSES = ["player_near", "player_far"]

# ================= 全局状态 =================
drawing = False
start_x, start_y = -1, -1
current_boxes = []  # 存储当前图片的框: [(class_id, x1, y1, x2, y2)]
img_list = []
current_img_index = 0
img_copy = None  # 用于绘制过程的图像副本


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_to_yolo_format(img_w, img_h, box):
    """将 (x1,y1,x2,y2) 转换为 YOLO 的 (x_center, y_center, w, h) 并归一化"""
    cls_id, x1, y1, x2, y2 = box
    dw = 1. / img_w
    dh = 1. / img_h
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return (cls_id, x_center * dw, y_center * dh, w * dw, h * dh)


def save_annotations(img_path, boxes, img_w, img_h):
    """保存当前图片的标注到 txt 文件 (Python自带的open函数支持中文路径)"""
    if not boxes:
        return

    img_name = os.path.basename(img_path)
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(LABEL_DIR, txt_name)

    # 确保保存 txt 时使用 utf-8 编码
    with open(txt_path, 'w', encoding='utf-8') as f:
        for box in boxes:
            yolo_box = convert_to_yolo_format(img_w, img_h, box)
            f.write(f"{yolo_box[0]} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {yolo_box[4]:.6f}\n")
    print(f"已保存标注: {txt_name}")


def load_annotations(img_path, img_w, img_h):
    """如果之前标过，尝试加载已有标注"""
    global current_boxes
    current_boxes = []
    img_name = os.path.basename(img_path)
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(LABEL_DIR, txt_name)

    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    x_c, y_c, w_norm, h_norm = map(float, parts[1:])
                    # 反归一化为坐标
                    w = w_norm * img_w
                    h = h_norm * img_h
                    x1 = int((x_c * img_w) - (w / 2))
                    y1 = int((y_c * img_h) - (h / 2))
                    x2 = int((x_c * img_w) + (w / 2))
                    y2 = int((y_c * img_h) + (h / 2))
                    current_boxes.append((cls_id, x1, y1, x2, y2))


def draw_boxes(img):
    """在图像上绘制已有的框"""
    for box in current_boxes:
        cls_id, x1, y1, x2, y2 = box
        color = (0, 255, 255) if cls_id == 0 else (255, 0, 255)  # 近处黄色，远处紫色
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, CLASSES[cls_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, current_boxes, img_copy

    img_original = param

    if event == cv2.EVENT_LBUTTONDOWN:
        # 如果已经画了两个框，不再响应
        if len(current_boxes) >= 2:
            print("已经标注了两个运动员！按 'C' 重新标注。")
            return

        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img_original.copy()
            draw_boxes(img_copy)
            # 绘制正在拉动的框
            cls_id = len(current_boxes)  # 0 为近，1 为远
            color = (0, 255, 255) if cls_id == 0 else (255, 0, 255)
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), color, 2)
            cv2.imshow("YOLO Annotator", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            # 修正坐标，防止反向拖动鼠标
            x1, x2 = min(start_x, x), max(start_x, x)
            y1, y2 = min(start_y, y), max(start_y, y)

            # 过滤太小的误触框
            if x2 - x1 > 5 and y2 - y1 > 5:
                cls_id = len(current_boxes)
                current_boxes.append((cls_id, x1, y1, x2, y2))

            img_copy = img_original.copy()
            draw_boxes(img_copy)
            cv2.imshow("YOLO Annotator", img_copy)


def main():
    global current_img_index, current_boxes, img_copy

    ensure_dir(LABEL_DIR)

    # 获取所有图片
    valid_exts = ('*.jpg', '*.jpeg', '*.png')
    img_paths = []
    for ext in valid_exts:
        img_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"错误: 在 {IMAGE_DIR} 文件夹下没有找到图片。请检查路径。")
        return

    cv2.namedWindow("YOLO Annotator", cv2.WINDOW_AUTOSIZE)

    while current_img_index < len(img_paths):
        img_path = img_paths[current_img_index]

        # ！！！关键修复！！！
        # 替换原有的 cv2.imread，使用 numpy 读取二进制流再用 cv2.imdecode 解码
        try:
            img_data = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"读取文件异常: {img_path}, 错误: {e}")
            img = None

        if img is None:
            print(f"无法解码图片，跳过: {os.path.basename(img_path)}")
            current_img_index += 1
            continue

        img_h, img_w = img.shape[:2]

        # 加载已有标注（如果有）
        load_annotations(img_path, img_w, img_h)

        img_copy = img.copy()
        draw_boxes(img_copy)

        # 绑定鼠标事件
        cv2.setMouseCallback("YOLO Annotator", mouse_callback, param=img)

        while True:
            # 增加 UI 提示文字
            display_img = img_copy.copy()
            text = f"Img {current_img_index + 1}/{len(img_paths)} | [Space/D]: Next | [A]: Prev | [C]: Clear Box | [X]: Del Img | [Q]: Quit"
            cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("YOLO Annotator", display_img)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('d') or key == 32:  # D键 或 空格: 下一张 (并保存)
                save_annotations(img_path, current_boxes, img_w, img_h)
                current_img_index += 1
                break

            elif key == ord('a'):  # A键: 上一张 (并保存当前)
                save_annotations(img_path, current_boxes, img_w, img_h)
                current_img_index = max(0, current_img_index - 1)
                break

            elif key == ord('c'):  # C键: 清除当前图片的所有框
                current_boxes = []
                img_copy = img.copy()
                print("已清空当前图片的标注框")

            elif key == ord('x'):  # X键: 删除当前图片及对应的标签
                try:
                    os.remove(img_path)
                    txt_path = os.path.join(LABEL_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                    print(f"已彻底删除废片: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"删除文件失败: {e}")

                img_paths.pop(current_img_index)
                if current_img_index >= len(img_paths):
                    current_img_index = len(img_paths) - 1
                break

            elif key == ord('q') or key == 27:  # Q键 或 ESC: 退出
                save_annotations(img_path, current_boxes, img_w, img_h)
                print("退出标注工具")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("所有图片浏览完毕！")


if __name__ == "__main__":
    main()