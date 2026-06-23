"""
批量评估：单进程+数据共享，3模型一组共享数据。
用法: cd 项目标注与测试 && .venv/Scripts/python src/utils/batch_eval_all.py
输出: ../论文/batch_eval_results.csv
"""
import sys, os, csv, glob, gc
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
_MST_DIR = os.path.join(_SRC_DIR, "model", "mst")
sys.path.insert(0, _MST_DIR)
from dataset import TennisActionDataset
from model_main import MSTFormer
from config import load_config

RESULTS_FILE = os.path.join(_PROJECT_DIR, "论文", "batch_eval_results.csv")

GROUPS = [
    [("main", "main"), ("optimal", "optimal"), ("sf_main", "sf_main")],
    [("abl_no_pose", "ablation/abl_no_pose"), ("abl_no_crops", "ablation/abl_no_crops"), ("abl_no_visual", "ablation/abl_no_visual")],
    [("abl_global_only", "ablation/abl_global_only"), ("cmp_ce_loss", "components/cmp_ce_loss"), ("cmp_focal_loss", "components/cmp_focal_loss")],
    [("cmp_no_merge", "components/cmp_no_merge"), ("cmp_resnet_backbone", "components/cmp_resnet_backbone"), ("cmp_frozen_backbone", "components/cmp_frozen_backbone")],
    [("hp_depth4", "hyperparams/hp_depth4"), ("hp_depth12", "hyperparams/hp_depth12"), ("hp_embed96", "hyperparams/hp_embed96")],
    [("hp_embed256", "hyperparams/hp_embed256"), ("hp_vtokens8", "hyperparams/hp_vtokens8"), ("hp_vtokens32", "hyperparams/hp_vtokens32")],
]

def try_load(name, yaml_sub):
    """尝试加载模型，失败返回None"""
    pattern = os.path.join(_PROJECT_DIR, "models", "action", name, "*", "best.pth")
    cands = glob.glob(pattern)
    if not cands: return None, name
    latest = max(cands, key=os.path.getmtime)
    ts_dir = os.path.dirname(latest)

    # 找配置文件
    cfg_path = None
    for fn in ["config.yaml", "config.json"]:
        fp = os.path.join(ts_dir, fn)
        if os.path.exists(fp): cfg_path = fp; break
    if not cfg_path:
        # 回退
        alt = os.path.join(_PROJECT_DIR, "configs", yaml_sub + ".yaml")
        if os.path.exists(alt): cfg_path = alt
    if not cfg_path: return None, name

    try:
        device = torch.device("cuda")
        raw_cfg = load_config(cfg_path)
        raw_cfg["data_root"] = os.path.join(_PROJECT_DIR, "data", "rallies_train")
        raw_cfg.pop("_smoke_clip", None)
        if raw_cfg.get("image_augment"): raw_cfg["image_augment"] = False
        raw_cfg["batch_size"] = 4
        raw_cfg.pop("test_data_root", None)

        model = MSTFormer(raw_cfg).to(device)
        sd = torch.load(latest, map_location=device)
        if "model_state_dict" in sd: sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        model.eval()
        return (model, raw_cfg, os.path.basename(ts_dir), name)
    except Exception as e:
        print(f"  {name} 加载失败: {e}")
        return None, name

def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    device = torch.device("cuda")
    data_root = os.path.join(_PROJECT_DIR, "data", "rallies_train")

    # 数据切分
    import random, cv2
    random.seed(42)
    clips = []
    for d in os.listdir(data_root):
        cp = os.path.join(data_root, d)
        if not os.path.isdir(cp): continue
        vp = os.path.join(cp, "raw_clip.mp4")
        ap = os.path.join(cp, "annotations.json")
        if not os.path.exists(vp) or not os.path.exists(ap): continue
        cap = cv2.VideoCapture(vp)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if nf > 0: clips.append((cp, nf))
    random.shuffle(clips)
    _, test_dirs = [], []
    cur = 0
    target = sum(n for _, n in clips) * 0.8
    for p, n in clips:
        if cur < target: cur += n
        else: test_dirs.append(p)
    print(f"测试集: {len(test_dirs)} clips")

    # DataLoader（只创建一次）
    base_cfg = {"data_root": data_root, "seq_len": 120, "min_seq_len": 60,
                "use_visual": True, "use_player_crops": True}
    ds = TennisActionDataset({**base_cfg, "num_classes": 5, "crops_root": data_root,
                               "reshuffle_augment": False}, clip_dirs=test_dirs, augment=False)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    results = []
    for gi, group in enumerate(GROUPS):
        print(f"\n{'='*60}")
        print(f"  组 {gi+1}/{len(GROUPS)}: {[g[0] for g in group]}")
        print(f"{'='*60}")

        # 加载模型
        loaded = []
        for name, yaml_sub in group:
            m = try_load(name, yaml_sub)
            if m[0] is not None:
                loaded.append(m)
                print(f"  {name}")
            else:
                print(f"  {name} 跳过: {m[1] if len(m)>1 else 'no model'}")

        if not loaded:
            for name, _ in group: results.append([name, "FAILED"])
            continue

        # 推理
        all_out = {m[3]: ([], []) for m in loaded}
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"  组{gi+1}"):
                pose, packed, labels, _kf = batch
                pose = pose.to(device, non_blocking=True)
                packed = packed.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                mask = labels != -100

                for model, cfg, ts, name in loaded:
                    with torch.amp.autocast("cuda"):
                        out = model(pose, packed)
                    preds = out[0].argmax(-1)
                    all_out[name][0].extend(preds[mask].cpu().numpy())
                    all_out[name][1].extend(labels[mask].cpu().numpy())

        # 计算指标并保存
        for model, cfg, ts, name in loaded:
            yt = np.array(all_out[name][1])
            yp = np.array(all_out[name][0])
            acc = (yt == yp).mean()
            results.append([name, f"{acc*100:.2f}"])

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(yt, yp, labels=[0,1,2,3,4])
            out_dir = os.path.join(_PROJECT_DIR, "models", "action", name, ts, "eval_full")
            os.makedirs(out_dir, exist_ok=True)
            np.savetxt(os.path.join(out_dir, "cm_test.csv"), cm, delimiter=",", fmt="%d")
            print(f"  ▶ {name}: test_acc={acc*100:.2f}%")

        # 释放
        loaded.clear()
        torch.cuda.empty_cache()
        gc.collect()

    # 写结果
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "test_acc"])
        for r in results:
            w.writerow(r)

    print(f"\n{'='*50}")
    print(f"  完成! 结果: {RESULTS_FILE}")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r[0]:<20} {r[1]}")

if __name__ == "__main__":
    main()
