"""
生成毕业论文图表（使用batch_eval_results.csv真实数据）。
运行：cd 项目标注与测试 && .venv/Scripts/python src/utils/generate_thesis_figures.py
"""
import os, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 设置中文字体（Windows 环境下使用 Microsoft YaHei）
for fname in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/msyhbd.ttc',
              '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']:
    if os.path.exists(fname):
        font_prop = fm.FontProperties(fname=fname)
        plt.rcParams['font.family'] = font_prop.get_name()
        break
else:
    # fallback: 尝试查找系统已注册的中文字体
    for f in fm.findSystemFonts():
        if 'yahei' in f.lower() or 'microhei' in f.lower() or 'simhei' in f.lower():
            plt.rcParams['font.family'] = fm.FontProperties(fname=f).get_name()
            break
plt.rcParams['axes.unicode_minus'] = False

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUTPUT_DIR = os.path.join(_PROJECT_DIR, "论文", "figures")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ─── 真实数据（来自batch_eval_results.csv）────────────────
DATA = {
    "main": 85.89, "optimal": 83.79, "sf_main": 86.55,
    "abl_no_pose": 85.48, "abl_no_crops": 76.53,
    "abl_no_visual": 68.22, "abl_global_only": 73.51,
    "cmp_ce_loss": 86.22, "cmp_focal_loss": 84.83,
    "cmp_no_merge": 76.51, "cmp_resnet_backbone": 82.47,
    "cmp_frozen_backbone": 68.22,
    "hp_depth4": 85.07, "hp_depth12": 84.18,
    "hp_embed96": 84.71, "hp_embed256": 86.18,
    "hp_vtokens8": 85.67, "hp_vtokens32": 83.74,
    "main_shared": 85.89,
}

C_MAIN, C_ABL, C_CMP, C_HP, C_BEST = '#2196F3', '#FF7043', '#AB47BC', '#66BB6A', '#FFD600'

def add_labels(bars, ax):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f'{h:.1f}%', ha='center', fontsize=8, fontweight='bold')

# ═══ fig1: 主模型训练曲线 ═════════════════════════════════
def fig1():
    path = os.path.join(_PROJECT_DIR, "models", "action", "main", "20260424_195300", "train_log.csv")
    ep, ta, tra, loss = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ep.append(int(row['epoch'])); ta.append(float(row['test_acc']))
            tra.append(float(row['train_acc'])); loss.append(float(row['train_loss']))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)', color=C_MAIN)
    ax1.plot(ep, ta, '-', color=C_MAIN, lw=2, label='Test Acc')
    ax1.plot(ep, tra, '--', color=C_MAIN, alpha=0.5, lw=1.5, label='Train Acc')
    ax1.tick_params(axis='y', labelcolor=C_MAIN); ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx(); ax2.set_ylabel('Loss', color=C_ABL)
    ax2.plot(ep, loss, '-', color=C_ABL, lw=1.5, label='Train Loss')
    ax2.tick_params(axis='y', labelcolor=C_ABL); ax2.legend(loc='upper right')
    plt.title('Main Model Training Curve (85.37% Test Acc)')
    fig.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, 'fig1_main_training_curve.png'), dpi=200, bbox_inches='tight'); plt.close()

# ═══ fig2: 消融实验 ═══════════════════════════════════════
def fig2():
    labels = ['Full', 'No Pose', 'No Crops', 'Pose\nOnly', 'Global\nOnly']
    vals = [DATA['main'], DATA['abl_no_pose'], DATA['abl_no_crops'],
            DATA['abl_no_visual'], DATA['abl_global_only']]
    colors = [C_MAIN] + [C_ABL]*4
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    add_labels(bars, ax)
    ax.set_ylabel('Test Accuracy (%)'); ax.set_title('Ablation Study')
    ax.set_ylim(60, 100)
    fig.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, 'fig2_ablation_comparison.png'), dpi=200, bbox_inches='tight'); plt.close()

# ═══ fig3: 组件对比 ═══════════════════════════════════════
def fig3():
    groups = [
        ('Loss Function', [('Focal\n(main)', DATA['main'], C_MAIN), ('CE Loss', DATA['cmp_ce_loss'], C_ABL), ('Focal Only', DATA['cmp_focal_loss'], C_ABL)]),
        ('Token Strategy', [('Merge\n(main)', DATA['main'], C_MAIN), ('Independent', DATA['cmp_no_merge'], C_ABL)]),
        ('Backbone', [('YOLO11\n(main)', DATA['main'], C_MAIN), ('ResNet18', DATA['cmp_resnet_backbone'], C_ABL)]),
        ('Backbone\nTraining', [('Unfreeze\n(main)', DATA['main'], C_MAIN), ('Frozen', DATA['cmp_frozen_backbone'], C_ABL)]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    for ax, (title, items) in zip(axes, groups):
        bars = ax.bar([x[0] for x in items], [x[1] for x in items], color=[x[2] for x in items], width=0.5)
        add_labels(bars, ax)
        ax.set_ylim(60, 100); ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', labelsize=7)
    fig.suptitle('Component Comparison Experiments', fontsize=13, y=1.02)
    fig.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, 'fig3_component_comparison.png'), dpi=200, bbox_inches='tight'); plt.close()

# ═══ fig4: 超参数 ════════════════════════════════════════
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # depth
    d_v = [DATA['hp_depth4'], DATA['main'], DATA['hp_depth12']]
    bars = axes[0].bar(['depth=4', 'depth=8', 'depth=12'], d_v, color=[C_HP, C_MAIN, C_ABL], width=0.5)
    add_labels(bars, axes[0]); axes[0].set_title('Transformer Depth'); axes[0].set_ylim(75, 100); axes[0].set_ylabel('Test Acc (%)')
    # embed_dim
    e_v = [DATA['hp_embed96'], DATA['main'], DATA['hp_embed256']]
    bars = axes[1].bar(['dim=96', 'dim=128', 'dim=256'], e_v, color=[C_ABL, C_MAIN, C_HP], width=0.5)
    add_labels(bars, axes[1]); axes[1].set_title('Embedding Dimension'); axes[1].set_ylim(75, 100)
    # visual_tokens
    v_v = [DATA['hp_vtokens8'], DATA['main'], DATA['hp_vtokens32']]
    bars = axes[2].bar(['vt=8', 'vt=16', 'vt=32'], v_v, color=[C_HP, C_MAIN, C_ABL], width=0.5)
    add_labels(bars, axes[2]); axes[2].set_title('Visual Tokens'); axes[2].set_ylim(75, 100)
    fig.suptitle('Hyperparameter Experiments', fontsize=13)
    fig.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, 'fig4_hyperparameter_comparison.png'), dpi=200, bbox_inches='tight'); plt.close()

# ═══ fig5: 类别分布 ═══════════════════════════════════════
def fig5():
    sizes = [56037, 4470, 4046, 6750, 16628]
    trimmed = [36314, 3692, 3285, 5795, 13524]
    names = ['待机\n(idle)', '正手\n(FH)', '反手\n(BH)', '发球\n(serve)', '移动\n(move)']
    cols = ['#9E9E9E', '#4CAF50', '#FF9800', '#F44336', '#2196F3']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, vals, title in [(ax1, sizes, 'Original'), (ax2, trimmed, 'After Trimming')]:
        wedges, _, autotexts = ax.pie(vals, labels=None, colors=cols, autopct='%1.1f%%', startangle=90)
        ax.set_title(title)
    fig.legend(names, loc='lower center', ncol=5, fontsize=9)
    fig.suptitle('Action Class Distribution', fontsize=13, y=1.02)
    fig.tight_layout(); plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(_OUTPUT_DIR, 'fig5_class_distribution.png'), dpi=200, bbox_inches='tight'); plt.close()

# ═══ fig6: 全部模型汇总 ═══════════════════════════════════
def fig6():
    items = [
        ('main\n(85.4)', DATA['main'], C_MAIN),
        ('sf_main\n(85.0)', DATA['sf_main'], '#00BCD4'),
        ('emb256\n(86.2)', DATA['hp_embed256'], C_BEST),
        ('no_pose', DATA['abl_no_pose'], C_ABL),
        ('no_crops', DATA['abl_no_crops'], C_ABL),
        ('no_visual', DATA['abl_no_visual'], C_ABL),
        ('global_only', DATA['abl_global_only'], C_ABL),
        ('CE Loss', DATA['cmp_ce_loss'], C_CMP),
        ('Focal', DATA['cmp_focal_loss'], C_CMP),
        ('no_merge', DATA['cmp_no_merge'], C_CMP),
        ('ResNet18', DATA['cmp_resnet_backbone'], C_CMP),
        ('frozen', DATA['cmp_frozen_backbone'], C_CMP),
        ('dp4', DATA['hp_depth4'], C_HP),
        ('dp12', DATA['hp_depth12'], C_HP),
        ('emb96', DATA['hp_embed96'], C_HP),
        ('vt8', DATA['hp_vtokens8'], C_HP),
        ('vt32', DATA['hp_vtokens32'], C_HP),
        ('shared\n(85.9)', DATA['main_shared'], '#FFD600'),
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(items)), [v for _,v,_ in items], color=[c for _,_,c in items], width=0.7)
    ax.axhline(y=85, color=C_BEST, linestyle='--', lw=1.5, alpha=0.7, label='85%')
    for bar, (l, v, _) in zip(bars, items):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{v:.1f}', ha='center', fontsize=7)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([l for l,_,_ in items], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Test Accuracy (%)'); ax.set_title('All Model Results'); ax.set_ylim(60, 100); ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, 'fig6_all_models_summary.png'), dpi=200, bbox_inches='tight'); plt.close()

# ═══ fig8: 关键帧F1 ═══════════════════════════════════════
def fig8():
    path = os.path.join(_PROJECT_DIR, "models", "action", "main", "20260424_195300", "train_log.csv")
    ep, kf = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ep.append(int(row['epoch']))
            kf.append(float(row['kf_f1']) if row['kf_f1'] else 0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(ep, kf, '-', color='#E91E63', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score (%)')
    ax.set_title('Keyframe Detection F1 During Training')
    fig.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, 'fig8_keyframe_curve.png'), dpi=200, bbox_inches='tight'); plt.close()

if __name__ == '__main__':
    print("生成论文图表（batch_eval真实数据）...")
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6, fig8]:
        fn()
    print(f"完成！{len(os.listdir(_OUTPUT_DIR))} 个图表已更新")
