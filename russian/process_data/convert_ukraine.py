"""
数据转换脚本：ukraine_opendialkg_pog_dialogs.json → JSONL 格式
参照 src/data/convert_opendialkg.py 的处理逻辑，适配实验室乌克兰数据集。

数据结构说明（每条对话固定 8 个 turn）：
  turn0: user  chat         —— 原始 OSINT 问题
  turn1: assistant action   —— 子问题1 对应的 KG 路径
  turn2: assistant chat     —— 子问题1 的回答
  turn3: assistant action   —— 子问题2 对应的 KG 路径
  turn4: assistant chat     —— 子问题2 的回答
  turn5: assistant action   —— 子问题3 对应的 KG 路径
  turn6: assistant chat     —— 子问题3 的回答
  turn7: assistant chat     —— 最终综合评估（无 KG 路径）

输出 TXT 格式（每行一个 JSON 字符串，与 convert_opendialkg.py 完全一致）：
  {"history": [...], "response": [...], "knowledge_base": [...], "dialogue_id": "1"}

用法示例：
  python convert_ukraine.py \\
      --input_file  g:/小论文/data/opendialkg/data/ukraine_opendialkg_pog_dialogs.json \\
      --out_file    ./ukraine_train.txt \\
      --split       train          # train / valid / test / all
"""

import json
import re
import argparse
import random
from typing import Any, Dict, Iterable, List, Tuple
from tqdm import tqdm


# ─────────────────────────────────────────────
# 文本规范化（与原脚本保持一致）
# ─────────────────────────────────────────────
def process(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\b)(D|d)(o)(es)?(nt)(\b)", r"\1\2\3\4n't\6", text)
    text = re.sub(r"(\b)(D|d)(idnt)(\b)",        r"\1\2idn't\4",   text)
    text = re.sub(r"(\b)(C|c)(ant)(\b)",          r"\1\2an't\4",    text)
    text = re.sub(r"(\b)(A|a)(rent)(\b)",         r"\1\2ren't\4",   text)
    text = re.sub(r"(\b)(i)(\b)",                 r"\1I\3",         text)
    text = re.sub(r"(\b)(I|i)(snt)(\b)",          r"\1\2sn't\4",    text)
    text = re.sub(r"(\b)(w|W)(asnt)(\b)",         r"\1\2asn't\4",   text)
    text = re.sub(r"(\b)(w|W)(erent)(\b)",        r"\1\2eren't\4",  text)
    text = re.sub(r"(\b)(I|i)(m)(\b)",            r"\1I'm\4",       text)
    text = re.sub(r"(\b)(Ill)(\b)",               r"\1I'll\3",      text)
    text = re.sub(r"(\b)(I|i)(ve)(\b)",           r"\1I've\4",      text)
    text = re.sub(r"(\b)(ha)((ve)|s)(nt)(\b)",    r"\1ha\3n't\6",   text)
    text = re.sub(r"(\b)(DId)(\b)",               r"\1Did\3",       text)
    text = re.sub(r"(\b)(Iknow)(\b)",             r"\1I know\3",    text)
    return text.strip()


# ─────────────────────────────────────────────
# 单条对话解析（核心逻辑，对应原脚本 parse_message）
# ─────────────────────────────────────────────
def parse_dialog(dialog: List[Dict], dialogue_id: str) -> Iterable[Dict[str, Any]]:
    """
    遍历一条对话的所有 turn，按 action→chat 配对生成训练样本。

    逻辑与 convert_opendialkg.py 完全一致：
    - turn 0 (user chat)     : 加入 history
    - action turn            : 存储 knowledge_base（KG 路径）
    - assistant chat turn    : 生成一条样本，重置 knowledge_base，更新 history
    """
    history = []
    knowledge_base = []

    for i, turn in enumerate(dialog):
        if i == 0:
            # 第一轮：用户问题，加入对话历史
            if "message" in turn:
                history.append([turn["sender"], process(turn["message"])])
        else:
            if "metadata" in turn:
                # action turn：读取 KG 路径三元组
                path_data = turn["metadata"].get("path", [])
                if len(path_data) >= 2 and isinstance(path_data[1], list):
                    knowledge_base = [
                        triple for triple in path_data[1]
                        if isinstance(triple, list) and len(triple) == 3
                    ]
                else:
                    knowledge_base = []
            else:
                # chat turn（assistant 回复）：生成一条训练样本
                response_text = process(turn["message"])
                yield {
                    "history":        history.copy(),
                    "response":       [turn["sender"], response_text],
                    "knowledge_base": knowledge_base,
                    "dialogue_id":    dialogue_id,
                }
                # 重置 KG 路径，将本轮回复加入历史
                knowledge_base = []
                history.append([turn["sender"], response_text])


# ─────────────────────────────────────────────
# 读取 JSON 文件（对应原脚本 read_csv）
# ─────────────────────────────────────────────
def read_json(input_file: str) -> Iterable[Tuple[List[Dict], str]]:
    """
    读取 ukraine_opendialkg_pog_dialogs.json。
    文件结构：顶层为 list，每个元素是一条对话的 turn list。
    dialogue_id 用从 1 开始的整数字符串表示。
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for idx, dialog in enumerate(data, start=1):
        yield dialog, str(idx)


# ─────────────────────────────────────────────
# 主转换函数
# ─────────────────────────────────────────────
def convert(
    input_file: str,
    out_file: str,
    split: str = "all",
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed: int = 42,
):
    """
    split: "all"   —— 全量写入一个文件
           "train" / "valid" / "test" —— 按比例划分后写入对应分片
    """
    # 先收集所有样本
    all_samples = []
    for dialog, dialog_id in tqdm(read_json(input_file), desc="解析对话"):
        for sample in parse_dialog(dialog, dialog_id):
            all_samples.append(sample)

    total = len(all_samples)
    print(f"[统计] 对话总数: {total // 4}  生成样本总数: {total}")

    if split == "all":
        _write_txt(all_samples, out_file)
    else:
        # 按对话 ID 划分（以对话为单位，避免同一对话跨集）
        random.seed(seed)
        dialog_ids = list({s["dialogue_id"] for s in all_samples})
        random.shuffle(dialog_ids)

        n = len(dialog_ids)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_ids = set(dialog_ids[:n_train])
        val_ids   = set(dialog_ids[n_train:n_train + n_val])
        test_ids  = set(dialog_ids[n_train + n_val:])

        split_map = {"train": train_ids, "valid": val_ids, "test": test_ids}
        selected_ids = split_map[split]

        filtered = [s for s in all_samples if s["dialogue_id"] in selected_ids]
        print(f"[划分] {split}: {len(selected_ids)} 条对话 → {len(filtered)} 条样本")
        _write_txt(filtered, out_file)


def _write_txt(samples: List[Dict], out_file: str):
    with open(out_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"[写出] {len(samples)} 条样本 → {out_file}")


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="将 ukraine_opendialkg_pog_dialogs.json 转换为 RHO 训练用 JSONL"
    )
    parser.add_argument("--input_file",  type=str, required=True,
                        help="输入文件路径（ukraine_opendialkg_pog_dialogs.json）")
    parser.add_argument("--out_file",    type=str, required=True,
                        help="输出 JSONL 文件路径")
    parser.add_argument("--split",       type=str, default="all",
                        choices=["all", "train", "valid", "test"],
                        help="输出分片，all=全量，其余按 train_ratio/val_ratio 划分（默认 all）")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例（默认 0.8）")
    parser.add_argument("--val_ratio",   type=float, default=0.1,
                        help="验证集比例（默认 0.1，测试集=剩余）")
    parser.add_argument("--seed",        type=int, default=42,
                        help="随机种子（默认 42）")
    args = parser.parse_args()

    convert(
        input_file  = args.input_file,
        out_file    = args.out_file,
        split       = args.split,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()
