"""
全局统一引用编号脚本。
1. 收集所有章节的参考文献
2. 去重合并
3. 更新所有正文引用标记
用法: cd 项目标注与测试 && .venv/Scripts/python src/utils/unify_citations.py
"""
import re, os, sys

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PAPER_DIR = os.path.join(os.path.dirname(_PROJECT_DIR), "论文")  # 论文在项目外

CHAPTERS = [
    "第一章_绪论.md",
    "第二章_相关技术基础.md",
    "第三章_数据集构建与标注工具链.md",
    "第四章_数据预处理与多模态特征提取.md",
    "第五章_MSTFormer多流动作识别模型.md",
    "第六章_实验与分析.md",
    "第七章_系统集成与Demo应用.md",
    "第八章_总结与展望+参考文献.md",
]

def parse_references(text):
    """从正文中提取参考文献列表"""
    refs = []
    in_refs = False
    for line in text.split('\n'):
        if '## 参考文献' in line or '# 参考文献' in line:
            in_refs = True
            continue
        if in_refs:
            m = re.match(r'^\[(\d+)\]\s+(.+)', line)
            if m:
                refs.append((int(m.group(1)), m.group(2).strip()))
            elif line.strip() == '':
                pass
            elif not line.startswith('['):
                # 参考文献结束
                if refs:  # 只有已经收集到引用时才停止
                    break
    return refs

def normalize_ref(text):
    """标准化参考文献文本用于去重匹配"""
    t = text.lower().strip()
    # 去掉标点和空格
    t = re.sub(r'[.,:;\-\[\]\(\)""\'\']', '', t)
    t = re.sub(r'\s+', ' ', t)
    # 提取作者名+年份的前60个字符作为key
    return t[:80]

def main():
    all_chapter_refs = {}
    seen_refs = {}  # normalized_text -> (global_id, full_text)
    global_refs = []
    chapter_mappings = {}  # chapter -> {old_num -> global_num}

    # 1. 读取所有章节
    for ch in CHAPTERS:
        path = os.path.join(_PAPER_DIR, ch)
        if not os.path.exists(path):
            print(f"!!  {ch} 不存在，跳过")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        refs = parse_references(text)
        all_chapter_refs[ch] = (text, refs)
        print(f"[INFO] {ch}: {len(refs)} 条引用")

    # 2. 去重构建全局引用列表
    for ch, (text, refs) in all_chapter_refs.items():
        mapping = {}
        for old_num, full_text in refs:
            key = normalize_ref(full_text)
            if key in seen_refs:
                global_id, _ = seen_refs[key]
            else:
                global_id = len(global_refs) + 1
                global_refs.append((global_id, full_text))
                seen_refs[key] = (global_id, full_text)
            mapping[old_num] = global_id
        chapter_mappings[ch] = mapping

    print(f"\n[DATA] 全局去重后共 {len(global_refs)} 条引用")
    print(f"   去重前合计: {sum(len(r) for _,r in all_chapter_refs.values())} 条")
    print(f"   去重: {sum(len(r) for _,r in all_chapter_refs.values()) - len(global_refs)} 条\n")

    # 3. 更新所有正文引用标记并追加全局参考文献
    for ch, (text, refs) in all_chapter_refs.items():
        mapping = chapter_mappings[ch]

        # 替换正文中的所有 [N] 引用标记
        # 匹配 [数字] 但排除行首的 [数字]（那是参考文献条目本身）
        def replace_ref(m):
            num = int(m.group(1))
            if num in mapping:
                return f"[{mapping[num]}]"
            return m.group(0)

        # 只替换正文部分（去掉参考文献列表）
        parts = re.split(r'(## 参考文献|# 参考文献)', text)
        if len(parts) > 1:
            body = parts[0]
            ref_section = ''.join(parts[1:])
        else:
            body = text
            ref_section = ''

        # 替换正文中的引用
        new_body = re.sub(r'\[(\d+)\]', replace_ref, body)

        # 生成全局参考文献列表（仅包含本正文中用到的全局引用）
        used_global_ids = set(mapping.values())
        global_ref_text = '\n'.join(
            f'[{gid}] {text}' for gid, text in global_refs
            if gid in used_global_ids
        )

        # 替换或追加参考文献
        if ref_section:
            new_ref_section = '\n\n## 参考文献（全文统一编号）\n\n' + global_ref_text + '\n'
        else:
            new_ref_section = '\n\n## 参考文献（全文统一编号）\n\n' + global_ref_text + '\n'

        new_text = new_body + new_ref_section

        # 写回文件
        out_path = os.path.join(_PAPER_DIR, ch)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(new_text)

        print(f"[OK] {ch}: 引用更新完成 ({len(refs)}→{len(used_global_ids)} 条)")

    # 4. 打印全局引用列表
    print(f"\n{'='*60}")
    print(f"全局参考文献列表（{len(global_refs)} 条）")
    print(f"{'='*60}")
    for gid, text in global_refs:
        print(f"[{gid}] {text}")


if __name__ == "__main__":
    main()
