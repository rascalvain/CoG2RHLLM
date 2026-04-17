"""
数据处理脚本：从 ukraine_opendialkg_pog.json 中提取知识图谱三元组，
并生成 TransE 训练所需的全套文件，流程与 OpenKE_/process_data.py 保持一致。

输出文件（均写入 --output_dir）：
  ukraine_triples.txt      ── 所有三元组，tab 分隔，格式: head\trelation\ttail
  entity2id.txt            ── OpenKE 格式：第一行总数，后续 entity\tid
  relation2id.txt          ── OpenKE 格式：第一行总数，后续 relation\tid
  train2id.txt / valid2id.txt / test2id.txt
                           ── OpenKE 格式：第一行三元组数，后续 head_id\ttail_id\trel_id

用法示例：
  python process_data.py \
      --input_file  g:/小论文/data/opendialkg/data/ukraine_opendialkg_pog.json \
      --output_dir  ./ukraine_kg
"""

import json
import os
import re
import unicodedata
import argparse
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# 0. 字符串规范化：将 Unicode 转为最近似的 ASCII
#    保证 OpenKE C 底层库能正常解析实体/关系名
# ─────────────────────────────────────────────
def to_ascii(text: str) -> str:
    # 先用 NFKD 分解（如 é → e + combining accent），再只保留 ASCII 字符
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", errors="ignore").decode("ascii")
    # 压缩多余空白
    return re.sub(r"\s+", " ", ascii_text).strip()


# ─────────────────────────────────────────────
# 1. 从 JSON 中提取所有知识图谱三元组
# ─────────────────────────────────────────────
def extract_triples(input_file: str):
    """
    遍历每条对话中所有包含 metadata.path 字段的消息，读取 path[1] 内的三元组列表。
    path 结构示例：
        [0.0, [["Donald Trump", "IS_IDENTIFIED_AS", "President"]], "Donald Trump is President"]
    返回去重后的三元组集合 set((head, relation, tail))。
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_triples = set()
    for item in data:
        for msg in item.get("dialog", []):
            metadata = msg.get("metadata", {})
            path = metadata.get("path", [])
            if len(path) < 2:
                continue
            triples_in_path = path[1]
            for triple in triples_in_path:
                if len(triple) == 3:
                    h, r, t = [to_ascii(s.strip()) for s in triple]
                    if h and r and t:
                        all_triples.add((h, r, t))

    print(f"[提取] 唯一三元组: {len(all_triples)}  "
          f"(来自 {len(data)} 条样本)")
    return all_triples


# ─────────────────────────────────────────────
# 2. 写出原始三元组文本文件（tab 分隔）
# ─────────────────────────────────────────────
def write_raw_triples(triples, output_dir: str, filename="ukraine_triples.txt"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")
    print(f"[写出] 原始三元组 → {path}")
    return path


# ─────────────────────────────────────────────
# 3. 拆分训练 / 验证 / 测试集（与原脚本比例一致）
# ─────────────────────────────────────────────
def split_triples(triples_list, test_size=0.2, val_ratio=0.5, random_state=42):
    """
    train : val : test ≈ 80 : 10 : 10
    """
    train, test_val = train_test_split(
        triples_list, test_size=test_size, random_state=random_state
    )
    test, val = train_test_split(
        test_val, test_size=val_ratio, random_state=random_state
    )
    print(f"[划分] train={len(train)}  val={len(val)}  test={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────
# 4. 构建实体 / 关系词典，写出 *2id 文件
# ─────────────────────────────────────────────
def build_vocab(triples):
    """从三元组列表中收集实体和关系，统一转小写后建立映射。"""
    entity2id, relation2id = {}, {}
    for h, r, t in triples:
        h_l, r_l, t_l = h.lower(), r.lower(), t.lower()
        if h_l not in entity2id:
            entity2id[h_l] = len(entity2id)
        if t_l not in entity2id:
            entity2id[t_l] = len(entity2id)
        if r_l not in relation2id:
            relation2id[r_l] = len(relation2id)
    print(f"[词典] 实体数: {len(entity2id)}  关系数: {len(relation2id)}")
    return entity2id, relation2id


def write_vocab(vocab: dict, output_dir: str, filename: str):
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(len(vocab)) + "\n")
        for key, idx in vocab.items():
            f.write(f"{key}\t{idx}\n")
    print(f"[写出] {filename} → {path}")


# ─────────────────────────────────────────────
# 5. 写出 OpenKE 格式的 split2id 文件
#    格式：首行三元组总数，后续每行 head_id\ttail_id\trel_id
# ─────────────────────────────────────────────
def write_split2id(triples_list, entity2id: dict, relation2id: dict,
                   output_dir: str, split_name: str):
    """
    跳过词典中找不到的三元组（理论上不会出现，保留容错）。
    """
    outlines = []
    skipped = 0
    for h, r, t in triples_list:
        h_l, r_l, t_l = h.lower(), r.lower(), t.lower()
        if h_l not in entity2id or t_l not in entity2id or r_l not in relation2id:
            skipped += 1
            continue
        head_id = entity2id[h_l]
        tail_id = entity2id[t_l]
        rel_id  = relation2id[r_l]
        outlines.append(f"{head_id}\t{tail_id}\t{rel_id}\n")

    if skipped:
        print(f"[警告] {split_name}: 跳过 {skipped} 条无法映射的三元组")

    path = os.path.join(output_dir, f"{split_name}2id.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(len(outlines)) + "\n")
        for line in outlines:
            f.write(line)
    print(f"[写出] {split_name}2id.txt ({len(outlines)} 条) → {path}")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="从 ukraine_opendialkg_pog.json 生成 TransE 训练所需数据文件"
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="JSON 数据文件路径，例如 g:/小论文/data/opendialkg/data/ukraine_opendialkg_pog.json"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出目录，例如 ./ukraine_kg"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="验证+测试集占全量比例（默认 0.2）"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.5,
        help="在 test_size 中验证集占比（默认 0.5，即验证=测试=各 10%%）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: 提取三元组
    triples_set = extract_triples(args.input_file)
    triples_list = list(triples_set)

    # Step 2: 写出原始三元组文本
    write_raw_triples(triples_set, args.output_dir)

    # Step 3: 划分数据集
    train_triples, val_triples, test_triples = split_triples(
        triples_list,
        test_size=args.test_size,
        val_ratio=args.val_ratio,
        random_state=args.seed,
    )

    # Step 4: 词典仅由全量三元组构建，保证 split2id 不会缺失
    entity2id, relation2id = build_vocab(triples_list)
    write_vocab(entity2id,   args.output_dir, "entity2id.txt")
    write_vocab(relation2id, args.output_dir, "relation2id.txt")

    # Step 5: 写出各 split 的 id 文件
    for split_name, split_data in [("train", train_triples),
                                   ("valid", val_triples),
                                   ("test",  test_triples)]:
        write_split2id(split_data, entity2id, relation2id,
                       args.output_dir, split_name)

    print("\n[完成] 所有文件已写入:", os.path.abspath(args.output_dir))
    print("  entity2id.txt    — 实体ID映射")
    print("  relation2id.txt  — 关系ID映射")
    print("  train2id.txt     — 训练集 (head_id, tail_id, rel_id)")
    print("  valid2id.txt     — 验证集")
    print("  test2id.txt      — 测试集")
    print("  ukraine_triples.txt — 原始三元组文本（备用）")
    print("\n可直接将 --output_dir 作为 train_transe.py 的 --datadir 参数使用。")


if __name__ == "__main__":
    main()
