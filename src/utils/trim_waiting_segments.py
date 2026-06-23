"""
修剪数据集中的等待段，减少类别不平衡。
从 data/rallies_train/ 读取，处理后输出到 data/rallies_train_trimmed/。

三种方法（独立概率，可叠加，不修改原始数据）：
  方法1 (75%)：末尾等待最多保留2s。连续等待时只裁最后一个
  方法2 (70%)：两次发球间的等待段，发球1后留1.5s，发球2前留1s
  方法3 (50%)：开头等待只保留最后2s
"""

import json
import random
import shutil
import os
import sys
from pathlib import Path

# ─── 配置 ───────────────────────────────────────────
SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "rallies_train"
DST_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "rallies_train_trimmed"
SEED = 42
PROB_M1 = 0.75   # 末尾等待 → 最多2s
PROB_M2 = 0.70   # 两次发球间等待裁切
PROB_M3 = 0.50   # 开头等待 → 最后2s

ACTION_WAIT = 0
ACTION_SERVE = 3

# ─── I/O ─────────────────────────────────────────────


def load_annotations(rally_path):
    ann_path = os.path.join(rally_path, "annotations.json")
    if not os.path.exists(ann_path):
        return None
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_annotations(annotations, dst_path):
    ann_path = os.path.join(dst_path, "annotations.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


# ─── 条件检查 ────────────────────────────────────────


def count_trailing_wait(annots):
    """统计末尾连续等待段的数量（=0 表示末尾不是等待）"""
    cnt = 0
    for a in reversed(annots):
        if a["action_id"] == ACTION_WAIT:
            cnt += 1
        else:
            break
    return cnt


def has_two_serves(annots):
    return sum(1 for a in annots if a["action_id"] == ACTION_SERVE) >= 2


def has_leading_wait(annots):
    return bool(annots) and annots[0]["action_id"] == ACTION_WAIT


# ─── 方法1：修剪末尾等待 ─────────────────────────────


def trim_trailing_wait(annots):
    """末尾等待段最多保留2s。连续等待时只裁最后一个。"""
    consecutive = count_trailing_wait(annots)
    if consecutive == 0:
        return annots

    # 只处理最后一个等待段
    last = annots[-1]
    duration = last["end_time"] - last["start_time"]
    if duration > 2.0:
        last["end_time"] = round(last["start_time"] + 2.0, 3)
    return annots


# ─── 方法2：两次发球间的等待 ─────────────────────────


def trim_between_serves(annots):
    """两次发球间的等待段：
    - 发球1结束后 1.5s 内的等待保留
    - 发球2开始前 1s 内的等待保留
    - 其他等待段全部移除
    """
    serve_indices = [i for i, a in enumerate(annots) if a["action_id"] == ACTION_SERVE]
    if len(serve_indices) < 2:
        return annots

    idx1, idx2 = serve_indices[0], serve_indices[1]
    serve1_end = annots[idx1]["end_time"]
    serve2_start = annots[idx2]["start_time"]

    keep_end = serve1_end + 1.5    # 发球1后保留到
    keep_start = serve2_start - 1.0  # 发球2前从这开始保留

    # 分两阶段：
    # 1) 收集所有需要做的事（记录，不动原列表）
    # 2) 从后往前执行（避免索引偏移）

    actions = []  # (index_in_annots, 'remove') or (index, 'modify', new_start, new_end)
    new_segs_before_serve2 = []  # dicts to insert before serve2

    between_indices = list(range(idx1 + 1, idx2))
    for i in between_indices:
        s = annots[i]
        if s["action_id"] != ACTION_WAIT:
            continue

        ws, we = s["start_time"], s["end_time"]

        # 与保留区域的重叠情况
        in_keep1 = ws < keep_end    # 在发球1后1.5s区域内
        in_keep2 = we > keep_start  # 在发球2前1s区域内

        if not in_keep1 and not in_keep2:
            actions.append((i, "remove"))

        elif in_keep1 and in_keep2:
            if keep_end >= keep_start:
                # 两个保留区重叠，取交集
                new_s = max(ws, keep_start)
                new_e = min(we, keep_end)
                if new_s < new_e:
                    actions.append((i, "modify", new_s, new_e))
                else:
                    actions.append((i, "remove"))
            else:
                # 不重叠，拆成两段
                # 前半段：[ws, keep_end]
                if keep_end > ws:
                    actions.append((i, "modify", ws, keep_end))
                else:
                    actions.append((i, "remove"))
                # 后半段：[keep_start, we]
                if we > keep_start:
                    new_segs_before_serve2.append({
                        "start_time": round(keep_start, 3),
                        "end_time": round(we, 3),
                        "action_name": s["action_name"],
                        "action_id": s["action_id"]
                    })

        elif in_keep1:
            new_e = min(we, keep_end)
            actions.append((i, "modify", ws, new_e))

        else:  # in_keep2 only
            new_s = max(ws, keep_start)
            actions.append((i, "modify", new_s, we))

    # 执行：从后往前
    actions.sort(key=lambda x: x[0], reverse=True)
    for act in actions:
        idx = act[0]
        if act[1] == "remove":
            annots.pop(idx)
        else:
            _, _, new_s, new_e = act
            annots[idx]["start_time"] = round(new_s, 3)
            annots[idx]["end_time"] = round(new_e, 3)

    # 插入拆分出的新段（在 serve2 之前）
    # 找到当前 serve2 的位置
    cur_serve_indices = [i for i, a in enumerate(annots) if a["action_id"] == ACTION_SERVE]
    if len(cur_serve_indices) >= 2:
        pos = cur_serve_indices[1]
        for seg in new_segs_before_serve2:
            annots.insert(pos, seg)
            pos += 1

    return annots


# ─── 方法3：修剪开头等待 ─────────────────────────────


def trim_leading_wait(annots):
    """开头等待段只保留最后2s"""
    if not annots or annots[0]["action_id"] != ACTION_WAIT:
        return annots
    first = annots[0]
    duration = first["end_time"] - first["start_time"]
    if duration > 2.0:
        first["start_time"] = round(first["end_time"] - 2.0, 3)
    return annots


# ─── 复制与处理 ──────────────────────────────────────


def copy_with_hardlinks(src, dst):
    """用硬链接复制目录（节省磁盘空间），排除 annotations.json。"""
    if dst.exists():
        shutil.rmtree(dst)

    def _ignore(src_dir, names):
        return {"annotations.json"}

    shutil.copytree(src, dst, ignore=_ignore, copy_function=os.link,
                    dirs_exist_ok=True)


def process_rally(src_path, dst_path, rng):
    """加载注解 → 按概率应用三种方法 → 保存修改后的注解。"""
    raw_annots = load_annotations(src_path)
    if raw_annots is None:
        return []

    annots = json.loads(json.dumps(raw_annots))  # 深拷贝
    applied = []

    # 方法1
    if count_trailing_wait(annots) > 0 and rng.random() < PROB_M1:
        annots = trim_trailing_wait(annots)
        applied.append("m1")

    # 方法2
    if has_two_serves(annots) and rng.random() < PROB_M2:
        annots = trim_between_serves(annots)
        applied.append("m2")

    # 方法3
    if has_leading_wait(annots) and rng.random() < PROB_M3:
        annots = trim_leading_wait(annots)
        applied.append("m3")

    # 只要注解有变化就保存
    if applied:
        save_annotations(annots, dst_path)

    return applied


# ─── 主流程 ──────────────────────────────────────────


def main():
    if not SRC_ROOT.exists():
        print(f"错误：源目录不存在 {SRC_ROOT}")
        sys.exit(1)

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    rally_dirs = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()])
    total = len(rally_dirs)
    rng = random.Random(SEED)

    print(f"源目录: {SRC_ROOT}")
    print(f"目标目录: {DST_ROOT}")
    print(f"总片段数: {total}")
    print(f"方法1(末尾等待→2s): P={PROB_M1*100:.0f}%")
    print(f"方法2(发球间等待):  P={PROB_M2*100:.0f}%")
    print(f"方法3(开头等待→2s): P={PROB_M3*100:.0f}%")
    print(f"随机种子: {SEED}")
    print(f"数据文件: 硬链接（不占用额外空间）")
    print()

    stats = {"m1": 0, "m2": 0, "m3": 0, "any": 0, "skipped_no_ann": 0}

    for i, src in enumerate(rally_dirs):
        dst = DST_ROOT / src.name
        indent = " " * 4
        print(f"[{i+1}/{total}] {src.name}")

        ann_path = src / "annotations.json"
        if not ann_path.exists():
            print(f"{indent}跳过（无 annotations.json）")
            stats["skipped_no_ann"] += 1
            continue

        # 硬链接复制数据文件
        copy_with_hardlinks(src, dst)

        # 按概率处理注解
        local_rng = random.Random(SEED + i)
        applied = process_rally(src, dst, local_rng)

        if applied:
            print(f"{indent}已处理: {' + '.join(applied)}")
            stats["any"] += 1
            for m in applied:
                stats[m] += 1
        else:
            print(f"{indent}未命中（随机未抽中或不满足条件）")

    print()
    print("=" * 50)
    print("处理统计:")
    print(f"  总片段:             {total}")
    print(f"  有注解:             {total - stats['skipped_no_ann']}")
    print(f"  已修改（至少1种方法）: {stats['any']}")
    print(f"  方法1(末尾等待→2s):  {stats['m1']}")
    print(f"  方法2(发球间等待):   {stats['m2']}")
    print(f"  方法3(开头等待→2s):  {stats['m3']}")
    print(f"  跳过(无注解):        {stats['skipped_no_ann']}")
    print(f"  输出目录:           {DST_ROOT}")
    print("=" * 50)


if __name__ == "__main__":
    main()
