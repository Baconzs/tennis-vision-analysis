"""run_ablation.py — 依次串行训练所有实验配置（不含 main）

用法:
python src/model/mst/run_ablation.py              # 训练全部
python src/model/mst/run_ablation.py --group ablation
python src/model/mst/run_ablation.py --dry-run
python src/model/mst/run_ablation.py --smoke
"""
import os
import sys
import subprocess
import argparse
import time

_mst_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(os.path.dirname(os.path.dirname(_mst_dir)))

# (group, config_relative_path)
CONFIGS = [
    # --- 超参对比 ---
    ("hyperparams", "configs/hyperparams/hp_embed96.yaml"),
    ("hyperparams", "configs/hyperparams/hp_embed256.yaml"),
    ("hyperparams", "configs/hyperparams/hp_depth4.yaml"),
    ("hyperparams", "configs/hyperparams/hp_depth12.yaml"),
    ("hyperparams", "configs/hyperparams/hp_vtokens8.yaml"),
    ("hyperparams", "configs/hyperparams/hp_vtokens32.yaml"),
    # --- 消融实验 ---
    ("ablation",    "configs/ablation/abl_no_pose.yaml"),
    ("ablation",    "configs/ablation/abl_no_crops.yaml"),
    ("ablation",    "configs/ablation/abl_no_visual.yaml"),
    ("ablation",    "configs/ablation/abl_global_only.yaml"),
    # --- 组件对比 ---
    ("components",  "configs/components/cmp_focal_loss.yaml"),
    ("components",  "configs/components/cmp_ce_loss.yaml"),
    ("components",  "configs/components/cmp_no_merge.yaml"),
    ("components",  "configs/components/cmp_resnet_backbone.yaml"),
    ("components",  "configs/components/cmp_frozen_backbone.yaml"),
]

TRAIN_SCRIPT = os.path.join(_mst_dir, "train.py")
PYTHON = sys.executable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不实际运行")
    parser.add_argument("--smoke",   action="store_true", help="每个配置只跑 1 片段 1 轮，验证流程")
    parser.add_argument("--group",   default=None,
                        choices=["hyperparams", "ablation", "components"],
                        help="只训练指定分组")
    args = parser.parse_args()

    selected = [(g, c) for g, c in CONFIGS if args.group is None or g == args.group]

    print(f"共 {len(selected)} 个配置待训练" + (f"（分组: {args.group}）" if args.group else ""))

    results = []
    for idx, (group, cfg_rel) in enumerate(selected, 1):
        cfg_path = os.path.join(_project_dir, cfg_rel)
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        cmd = [PYTHON, TRAIN_SCRIPT, "--config", cfg_path]
        if args.smoke:
            cmd.append("--smoke")

        print(f"\n{'='*60}")
        print(f"▶ [{idx}/{len(selected)}] {group}/{name}")
        print(f"  命令: {' '.join(cmd)}")
        print(f"{'='*60}")

        if args.dry_run:
            results.append((group, name, "dry-run", 0))
            continue

        t0 = time.time()
        ret = subprocess.run(cmd, cwd=_project_dir)
        elapsed = time.time() - t0

        status = "成功" if ret.returncode == 0 else f"失败(code={ret.returncode})"
        results.append((group, name, status, elapsed))
        print(f"\n{status}  耗时 {elapsed/60:.1f} 分钟")

    print(f"\n{'='*60}")
    print("训练汇总:")
    for group, name, status, elapsed in results:
        t = f"{elapsed/60:.1f} min" if elapsed else ""
        print(f"  {status}  [{group}] {name}  {t}")


if __name__ == "__main__":
    main()
