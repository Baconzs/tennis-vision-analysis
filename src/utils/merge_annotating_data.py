"""
将 data/rallies_annotating/ 中的新标注数据合并到 data/rallies_annotated/。
tracking_data.json → pose_data.json（含 court 字段）
id=1 → near_player，id=2 → far_player
rally 编号从 rallies_annotated 中最大编号+1 开始续编。
"""
import os
import json
import shutil
import re

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(_PROJECT_DIR, "data", "rallies_annotating")
DST_ROOT = os.path.join(_PROJECT_DIR, "data", "rallies_annotated")


def _get_max_rally_num(dst_root):
    max_num = 0
    for name in os.listdir(dst_root):
        m = re.match(r"rally_(\d+)_", name)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num


def _convert_tracking_to_pose(tracking_path):
    """将 tracking_data.json 转换为 pose_data.json 格式（列表，按 frame_id 索引）。"""
    with open(tracking_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        return []

    total = max(fr["frame_id"] for fr in frames) + 1
    pose_list = [None] * total

    for fr in frames:
        fid = fr["frame_id"]
        court = fr.get("court", [])
        players = {p["id"]: p for p in fr.get("players", [])}

        entry = {"frame": fid, "court": court}

        for role, pid in [("near_player", 1), ("far_player", 2)]:
            p = players.get(pid)
            if p:
                entry[role] = {
                    "bbox": p["bbox"],
                    "keypoints": p["pose"],
                }
            else:
                entry[role] = None

        pose_list[fid] = entry

    # 填充空帧（frame_id 不连续时）
    result = []
    for i, entry in enumerate(pose_list):
        if entry is None:
            result.append({"frame": i, "court": [], "near_player": None, "far_player": None})
        else:
            result.append(entry)

    return result


def main():
    next_num = _get_max_rally_num(DST_ROOT) + 1
    print(f"rallies_annotated 最大编号: {next_num - 1}，新 rally 从 {next_num} 开始编号")

    copied = skipped = failed = 0

    match_dirs = sorted(
        d for d in os.listdir(SRC_ROOT)
        if os.path.isdir(os.path.join(SRC_ROOT, d)) and not d.startswith("_")
    )

    for match_dir in match_dirs:
        match_path = os.path.join(SRC_ROOT, match_dir)
        rally_dirs = sorted(
            d for d in os.listdir(match_path)
            if os.path.isdir(os.path.join(match_path, d))
        )

        for rally_dir in rally_dirs:
            src_rally = os.path.join(match_path, rally_dir)

            # 提取时长后缀
            m = re.search(r"(\d+\.\d+s)$", rally_dir)
            duration = m.group(1) if m else "0.0s"

            # 检查必要文件
            tracking_path = os.path.join(src_rally, "tracking_data.json")
            anno_path = os.path.join(src_rally, "annotations.json")
            video_path = os.path.join(src_rally, "raw_clip.mp4")

            if not all(os.path.exists(p) for p in [tracking_path, anno_path, video_path]):
                print(f"  [SKIP] {match_dir}/{rally_dir} — 缺少必要文件")
                skipped += 1
                continue

            new_name = f"rally_{next_num:03d}_{duration}"
            dst_rally = os.path.join(DST_ROOT, new_name)

            if os.path.exists(dst_rally):
                print(f"  [SKIP] {new_name} — 目标已存在")
                skipped += 1
                next_num += 1
                continue

            try:
                os.makedirs(dst_rally, exist_ok=True)

                # 转换 pose 数据
                pose_data = _convert_tracking_to_pose(tracking_path)
                pose_out = os.path.join(dst_rally, "pose_data.json")
                with open(pose_out, "w", encoding="utf-8") as f:
                    json.dump(pose_data, f, ensure_ascii=False)

                # 复制其他文件
                shutil.copy2(anno_path, os.path.join(dst_rally, "annotations.json"))
                shutil.copy2(video_path, os.path.join(dst_rally, "raw_clip.mp4"))

                print(f"  [COPY] {match_dir}/{rally_dir} → {new_name}")
                copied += 1
                next_num += 1

            except Exception as e:
                print(f"  [ERR]  {match_dir}/{rally_dir}: {e}")
                failed += 1

    print(f"\n完成：复制 {copied} 个，跳过 {skipped} 个，失败 {failed} 个")
    print(f"rallies_annotated 下一个可用编号: {next_num}")


if __name__ == "__main__":
    main()
