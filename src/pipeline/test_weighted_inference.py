"""test_weighted_inference.py — 加权推理测试脚本

功能：用 OpenCV 追踪器测试视频中的目标追踪效果，验证推理流程
"""
import os
import cv2
import numpy as np

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_PIPELINE_DIR))

# 视频捕获（替换为你的视频文件）
cap = cv2.VideoCapture(os.path.join(_PROJECT_DIR, "data", "rallies_annotated", "rally_003_24.0s", "raw_clip.mp4"))

# 用于存储首次检测到目标并初始化追踪器的 BBox
bbox = None
init_tracking = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 如果尚未初始化追踪器，尝试通过颜色检测初始化
    if not init_tracking:
        # 1. 预处理：模糊降低噪声 (可选)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        # 2. 转换色彩空间到 HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 3. 定义黑色的 HSV 上下限 (需根据具体光照反复调整)
        # H: 色调, S: 饱和度, V: 亮度
        # 极低 H,S,V 值通常代表黑色。这是一个大概范围：
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])  # S 可设大些兼容杂色，V 控制亮度，需极小

        # 4. 创建掩膜
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # 5. 形态学操作 (可选，如腐蚀、膨胀) 进一步降噪和连接小斑点
        # kernel = np.ones((3,3), np.uint8)
        # mask = cv2.erode(mask, kernel, iterations=1)
        # mask = cv2.dilate(mask, kernel, iterations=1)

        # 6. 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 查找面积最大的黑色小斑点 (假设就是远端运动员)
        max_contour = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 过滤太小的噪声斑点，但也不能太大 (因为运动员在远端很小)
            if 5 < area < 100 and area > max_area:  # 这里需要根据实际像素大小调整
                max_area = area
                max_contour = cnt

        if max_contour is not None:
            # 7. 获取初始 BBox
            x, y, w, h = cv2.boundingRect(max_contour)
            bbox = (x, y, w, h)

            # 8. 初始化追踪器
            success = tracker.init(frame, bbox)
            if success:
                init_tracking = True
                print("追踪已初始化，检测到黑色斑点:", bbox)
            else:
                print("追踪初始化失败")

    # 如果已初始化，更新追踪器位置
    else:
        success, bbox = tracker.update(frame)
        if success:
            # 绘制追踪框
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # 追踪丢失，可尝试重新颜色检测初始化 (逻辑类似上面)
            print("追踪丢失，尝试重新检测")
            init_tracking = False  # 重置状态以便下次尝试重新检测初始化

    # 显示结果
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()