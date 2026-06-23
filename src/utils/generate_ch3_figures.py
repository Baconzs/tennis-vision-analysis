"""生成第三章配图：数据可视化（球场关键点/姿态骨架/检测效果）"""
import json, cv2, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "data/rallies_train/rally_001_19.8s"
OUT = "论文/figures"
os.makedirs(OUT, exist_ok=True)

# 加载数据
with open(f"{BASE}/pose_data.json") as f:
    data = json.load(f)

# COCO 骨架连接定义
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]
COCO_COLORS = [
    (255,0,0),(255,0,0),(255,0,0),(255,0,0),  # 鼻子/眼/耳
    (0,255,0),(0,255,0),(0,255,0),(0,255,0),  # 肩/肘/腕
    (255,255,0),(255,255,0),  # 髋
    (0,0,255),(0,0,255),(0,0,255),(0,0,255),  # 膝/踝
]
COURT_COLORS = [(0,255,255)] * 14

def draw_skeleton(img, kps, color=(0,255,0), conf_thresh=0.1):
    h, w = img.shape[:2]
    pts = []
    for i, kp in enumerate(kps):
        x, y, c = kp
        if c < conf_thresh:
            pts.append(None)
            continue
        px, py = int(x), int(y)
        cv2.circle(img, (px, py), 3, COCO_COLORS[i % len(COCO_COLORS)], -1)
        pts.append((px, py))
    for i, j in SKELETON:
        if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
            cv2.line(img, pts[i], pts[j], color, 2)

def draw_court(img, court_kps):
    for i, kp in enumerate(court_kps):
        x, y, c = kp
        if c < 0.3:
            continue
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.putText(img, str(i), (int(x)+5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

# ═══ figA: 球场关键点检测示例 ═══
frame_idx = 10
f = data[frame_idx]
img_path = f"{BASE}/frames/{f['frame']:06d}.jpg"
img = cv2.imread(img_path)
if img is not None:
    h, w = img.shape[:2]
    draw_court(img, f['court'])
    cv2.imwrite(f"{OUT}/fig_court_keypoints.png", img)
    print(f"Court keypoints figure saved ({w}x{h})")
else:
    print(f"Cannot read {img_path}")

# ═══ figB: 姿态骨架叠加示例 ═══
img2 = cv2.imread(img_path)
if img2 is not None:
    # 画 bbox
    near = f['near_player']
    if near.get('bbox'):
        x1,y1,x2,y2 = [int(v) for v in near['bbox']]
        cv2.rectangle(img2, (x1,y1), (x2,y2), (255,255,0), 2)
        cv2.putText(img2, 'P1', (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    far = f['far_player']
    if far.get('bbox'):
        x1,y1,x2,y2 = [int(v) for v in far['bbox']]
        cv2.rectangle(img2, (x1,y1), (x2,y2), (255,0,255), 2)
        cv2.putText(img2, 'P2', (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    # 画骨架
    draw_skeleton(img2, f['near_player']['keypoints'], (255,255,0))
    draw_skeleton(img2, f['far_player']['keypoints'], (255,0,255))
    cv2.imwrite(f"{OUT}/fig_pose_overlay.png", img2)
    print("Pose overlay figure saved")

# ═══ figC: 球员裁剪图示例 ═══
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
for idx, (prefix, label) in enumerate([('player1', 'Near Player'), ('player2', 'Far Player')]):
    crop_path = f"{BASE}/{prefix}/{f['frame']:06d}.jpg"
    crop = cv2.imread(crop_path)
    if crop is not None:
        axes[idx].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(label, fontsize=10)
    axes[idx].axis('off')
fig.suptitle('Player Crops (320x320)', fontsize=11)
fig.tight_layout()
plt.savefig(f"{OUT}/fig_player_crops.png", dpi=150, bbox_inches='tight'); plt.close()
print("Player crops figure saved")

# ═══ figD: 难例挖掘效果柱状图 ═══
fig, ax = plt.subplots(figsize=(4, 3.5))
bars = ax.bar(['Initial', 'After HNM'], [55.0, 99.2], color=['#E74C3C', '#2ECC71'], width=0.5)
for bar, v in zip(bars, [55.0, 99.2]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{v}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('F1-Score (%)')
ax.set_title('Hard Negative Mining Effect')
ax.set_ylim(0, 110)
fig.tight_layout()
plt.savefig(f"{OUT}/fig_hnm_effect.png", dpi=150, bbox_inches='tight'); plt.close()
print("HNM bar chart saved")

print("\nAll figures saved to", OUT)
