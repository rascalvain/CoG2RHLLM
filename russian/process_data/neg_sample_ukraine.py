"""
负采样脚本：为乌克兰数据集训练集添加负样本
参照 negative_sampling_new/neg_sample_new.py 的处理逻辑，并做以下增强：

负样本分三个难度层级（优先级从高到低）：
  1. 硬负样本（hard）  ── 同一对话中其他轮次用过的 KB（上下文相关但配错了回复）
  2. 中负样本（medium）── 不同对话中共享至少一个实体的 KB（语义相近但来源不同）
  3. 易负样本（easy）  ── 全局随机选取的 KB（与原脚本行为一致）

处理规则：
  - knowledge_base 为空的样本（最终总结轮）：仅保留正样本，不做负采样
  - 所有正样本加 "type": 1，所有负样本加 "type": 0

用法示例：
  python neg_sample_ukraine.py \\
      --input_file  ./ukraine_train.txt \\
      --output_file ./ukraine_train_with_neg.txt \\
      --strategy    hard        # hard / medium / easy / mixed
      --neg_per_pos 1           # 每条正样本生成几条负样本
"""

import json
import random
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────
def load_data(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data


# ─────────────────────────────────────────────
# 构建辅助索引
# ─────────────────────────────────────────────
def build_indices(data: List[Dict]):
    """
    返回：
      kb_pool       : 所有非空 KB 列表（去重后的字符串化列表，用于随机采样）
      dialog2kbs    : {dialogue_id: [kb1, kb2, ...]} 同一对话的 KB 集合
      entity2kbs    : {entity_name: [kb1, kb2, ...]} 含某实体的 KB 集合
    """
    kb_pool = []
    kb_set_dedup = set()

    dialog2kbs: Dict[str, List] = defaultdict(list)
    entity2kbs: Dict[str, List] = defaultdict(list)

    for item in data:
        kb = item["knowledge_base"]
        if not kb:
            continue
        kb_key = json.dumps(kb, sort_keys=True)
        if kb_key not in kb_set_dedup:
            kb_set_dedup.add(kb_key)
            kb_pool.append(kb)

        # 同一对话的 KB
        dialog2kbs[item["dialogue_id"]].append(kb)

        # 按实体建索引
        for triple in kb:
            if isinstance(triple, list) and len(triple) == 3:
                entity2kbs[triple[0].lower()].append(kb)
                entity2kbs[triple[2].lower()].append(kb)

    print(f"[索引] 非空 KB 池大小（去重）: {len(kb_pool)}")
    return kb_pool, dialog2kbs, entity2kbs


# ─────────────────────────────────────────────
# 负样本选取策略
# ─────────────────────────────────────────────
def pick_hard_negative(
    current_kb: List, dialogue_id: str, dialog2kbs: Dict
) -> List | None:
    """同一对话中其他轮次的 KB（硬负样本）"""
    candidates = [
        kb for kb in dialog2kbs.get(dialogue_id, [])
        if kb != current_kb
    ]
    return random.choice(candidates) if candidates else None


def pick_medium_negative(
    current_kb: List, entity2kbs: Dict
) -> List | None:
    """共享至少一个实体但来源不同的 KB（中负样本）"""
    entities = set()
    for triple in current_kb:
        if isinstance(triple, list) and len(triple) == 3:
            entities.add(triple[0].lower())
            entities.add(triple[2].lower())

    candidates = []
    for ent in entities:
        for kb in entity2kbs.get(ent, []):
            if kb != current_kb:
                candidates.append(kb)

    return random.choice(candidates) if candidates else None


def pick_easy_negative(current_kb: List, kb_pool: List) -> List | None:
    """全局随机 KB（易负样本，与原脚本行为一致）"""
    candidates = [kb for kb in kb_pool if kb != current_kb]
    return random.choice(candidates) if candidates else None


def pick_negative(
    item: Dict,
    strategy: str,
    kb_pool: List,
    dialog2kbs: Dict,
    entity2kbs: Dict,
) -> List | None:
    """
    按策略选取负 KB：
      hard   ── 仅尝试硬负，失败返回 None
      medium ── 仅尝试中负，失败返回 None
      easy   ── 仅随机选取
      mixed  ── 依次尝试 hard → medium → easy
    """
    kb = item["knowledge_base"]
    did = item["dialogue_id"]

    if strategy == "hard":
        return pick_hard_negative(kb, did, dialog2kbs)
    elif strategy == "medium":
        return pick_medium_negative(kb, entity2kbs)
    elif strategy == "easy":
        return pick_easy_negative(kb, kb_pool)
    else:  # mixed
        neg = pick_hard_negative(kb, did, dialog2kbs)
        if neg is None:
            neg = pick_medium_negative(kb, entity2kbs)
        if neg is None:
            neg = pick_easy_negative(kb, kb_pool)
        return neg


# ─────────────────────────────────────────────
# 主处理：生成正 + 负样本对
# ─────────────────────────────────────────────
def create_samples(
    data: List[Dict],
    strategy: str,
    kb_pool: List,
    dialog2kbs: Dict,
    entity2kbs: Dict,
    neg_per_pos: int = 1,
    seed: int = 42,
) -> List[Dict]:
    random.seed(seed)

    output = []
    skipped_empty_kb = 0
    neg_failed = 0

    for item in data:
        # 正样本：加 type=1
        positive = dict(item, type=1)

        if not item["knowledge_base"]:
            # KB 为空（最终总结轮）：只写正样本，不做负采样
            output.append(positive)
            skipped_empty_kb += 1
            continue

        output.append(positive)

        # 负样本：替换 knowledge_base，type=0
        for _ in range(neg_per_pos):
            neg_kb = pick_negative(item, strategy, kb_pool, dialog2kbs, entity2kbs)
            if neg_kb is None:
                neg_failed += 1
                continue
            negative = {
                "history":        item["history"],
                "response":       item["response"],
                "knowledge_base": neg_kb,
                "dialogue_id":    item["dialogue_id"],
                "type":           0,
            }
            output.append(negative)

    print(f"[统计] 正样本: {sum(1 for s in output if s['type']==1)}")
    print(f"[统计] 负样本: {sum(1 for s in output if s['type']==0)}")
    print(f"[统计] KB为空跳过负采样: {skipped_empty_kb}")
    if neg_failed:
        print(f"[警告] 找不到合适负KB，跳过: {neg_failed} 次")
    return output


# ─────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────
def save_data(samples: List[Dict], output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"[写出] 共 {len(samples)} 条 → {output_file}")


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="乌克兰数据集负采样增量脚本")
    parser.add_argument("--input_file",  type=str, default="./ukraine_train.txt",
                        help="输入训练文件（convert_ukraine.py 生成的 .txt）")
    parser.add_argument("--output_file", type=str, default="./ukraine_train_with_neg.txt",
                        help="输出文件（正样本 + 负样本）")
    parser.add_argument("--strategy",   type=str, default="mixed",
                        choices=["hard", "medium", "easy", "mixed"],
                        help="负样本策略：hard/medium/easy/mixed（默认 mixed）")
    parser.add_argument("--neg_per_pos", type=int, default=1,
                        help="每条正样本生成的负样本数（默认 1）")
    parser.add_argument("--seed",        type=int, default=42,
                        help="随机种子（默认 42）")
    args = parser.parse_args()

    print(f"[配置] 策略={args.strategy}  每正样本负样本数={args.neg_per_pos}")

    # 加载数据
    data = load_data(args.input_file)
    print(f"[加载] 共 {len(data)} 条样本来自 {args.input_file}")

    # 构建索引
    kb_pool, dialog2kbs, entity2kbs = build_indices(data)

    # 生成正负样本
    samples = create_samples(
        data, args.strategy, kb_pool, dialog2kbs, entity2kbs,
        neg_per_pos=args.neg_per_pos, seed=args.seed
    )

    # 保存
    save_data(samples, args.output_file)


if __name__ == "__main__":
    main()
