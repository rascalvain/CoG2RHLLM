"""
测试/验证集构建脚本：适配乌克兰数据集
参照 src/data/build_dataset.py 的处理逻辑。

与训练集脚本 (build_dataset_ukraine.py) 的区别：
  1. 输入文件无 "type" 字段 → 统一默认为 1（全部正样本）
  2. 无负样本，不需要 --skip_neg 等标志
  3. filter_no_english 默认关闭（原脚本注释：test data 不能直接删去）
  4. 长度不一致时：打印警告并跳过，不 assert False 崩溃

输出 CSV 列：entity_relation_ids | memory_bank | history | response | type

用法示例：
  # 验证集
  python build_eval_ukraine.py \\
      --file_entity_id   ./ukraine_kg/entity2id.txt \\
      --file_relation_id ./ukraine_kg/relation2id.txt \\
      --input_file       ./ukraine_valid.txt \\
      --out_file         ./ukraine_valid_dataset.csv \\
      --bart_model       /root/autodl-fs/paper/RHO-main/bart_base \\
      --sbert_model      /root/autodl-fs/paper/RHO-main/paraphrase-distilroberta-base-v2

  # 测试集
  python build_eval_ukraine.py \\
      --file_entity_id   ./ukraine_kg/entity2id.txt \\
      --file_relation_id ./ukraine_kg/relation2id.txt \\
      --input_file       ./ukraine_test.txt \\
      --out_file         ./ukraine_test_dataset.csv \\
      --bart_model       /root/autodl-fs/paper/RHO-main/bart_base \\
      --sbert_model      /root/autodl-fs/paper/RHO-main/paraphrase-distilroberta-base-v2
"""

import os
import sys
import csv
import json
import re
import ast
import unicodedata
import argparse
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser


def to_ascii(text: str) -> str:
    """与 process_data.py 保持一致的 ASCII 规范化，确保实体名与 entity2id.txt 的键匹配"""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", errors="ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip()

# 配置 HuggingFace 国内镜像，同时设置新旧两个环境变量
# HF_ENDPOINT 对应新版 huggingface_hub（transformers/sentence-transformers 使用）
# HUGGINGFACE_CO_RESOLVE_ENDPOINT 对应旧版 huggingface_hub（Flair 内部使用）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer

# ── 定位 data_utils.py ──────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_UTILS_DIR = os.path.join(REPO_ROOT, "src", "data")
if DATA_UTILS_DIR not in sys.path:
    sys.path.insert(0, DATA_UTILS_DIR)

from data_utils import (
    Triple_id,
    match_entities,
    get_tokenized_idx,
    delete_response_part,
    only_english,
)
from sentence_transformers import SentenceTransformer
from flair.models import SequenceTagger


# ─────────────────────────────────────────────
# 核心构建函数
# ─────────────────────────────────────────────
def build_eval_dataset(
    tripleid,
    in_path: str,
    out_path: str,
    tokenizer,
    embedder,
    tagger,
    process_rel: str = "process_rel",
    mod: str = "all",
    max_hist_len: int = 3,
):
    """
    处理测试集 / 验证集。
    与训练集流程完全一致，差异：
      - type 字段缺失时默认为 1（eval 无负样本）
      - 不过滤非英文样本（test data 不能直接删去）
      - 长度不一致时警告跳过，不崩溃
    """
    pre = "Given the knowledge:"
    pre_len = len(tokenizer.tokenize(pre))

    skipped_empty  = 0
    skipped_assert = 0
    written        = 0

    with open(in_path, encoding="utf-8") as f_in, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:

        writer = csv.writer(fout)
        writer.writerow(["entity_relation_ids", "memory_bank", "history", "response", "type"])

        for idx, line in tqdm(enumerate(f_in.readlines()), desc="构建评估集"):
            if idx % 500 == 0:
                print(f"  处理第 {idx} 行...")

            l_in = json.loads(line)

            # ── 空 KB 处理 ──────────────────────────────
            if not l_in["knowledge_base"]:
                # eval 集保留空 KB 样本（对应最终总结轮）
                entity_relation_ids = [0] * pre_len
                memory_bank         = []
                input_text          = pre
                all_entities        = set()
                skipped_empty      += 1   # 仅计数，不跳过
            else:
                # ── 处理含 KB 样本 ────────────────────────
                entity_relation_ids = [0] * pre_len
                memory_bank         = []
                triples             = []
                all_entities        = set()

                for triple in l_in["knowledge_base"]:
                    sub, rel, obj = triple
                    # 与 process_data.py 保持一致：查字典前做 ASCII 规范化
                    sub = to_ascii(sub)
                    rel = to_ascii(rel)
                    obj = to_ascii(obj)
                    if not sub or not rel or not obj:
                        continue
                    all_entities.add(sub)
                    all_entities.add(obj)

                    if process_rel:
                        if "~" in rel:
                            rel = re.sub("~", "", rel)
                            sub, obj = obj, sub

                    try:
                        sub_id = tripleid.get_triple_id(sub, False)
                    except KeyError:
                        print(f"  [警告] 实体不在词典中，跳过: {triple}")
                        continue
                    try:
                        rel_id = tripleid.get_triple_id(rel, True)
                    except KeyError:
                        print(f"  [警告] 关系不在词典中，跳过: {triple}")
                        continue
                    rel_text = re.sub("[-_]", " ", rel)
                    try:
                        obj_id = tripleid.get_triple_id(obj, False)
                    except KeyError:
                        print(f"  [警告] 实体不在词典中，跳过: {triple}")
                        continue

                    memory_bank.append([sub_id, rel_id, obj_id])

                    sub_len = len(tokenizer.tokenize(" " + sub))
                    rel_len = len(tokenizer.tokenize(" " + rel_text))
                    obj_len = len(tokenizer.tokenize(" " + obj))

                    if mod == "all":
                        entity_relation_ids += (
                            [sub_id] * sub_len + [0]
                            + [rel_id] * rel_len + [0]
                            + [obj_id] * obj_len + [0]
                        )
                    elif mod == "first":
                        entity_relation_ids += (
                            [sub_id] + [0] * sub_len
                            + [rel_id] + [0] * rel_len
                            + [obj_id] + [0] * obj_len
                        )
                    else:
                        raise ValueError(f"不支持的 mod: {mod}")

                    triples.append(sub + "<sep> " + rel_text + "<sep> " + obj)

                entity_relation_ids = entity_relation_ids[:-1]
                input_text = pre + " " + "<triple> ".join(triples)

            # ── 构建对话历史文本 ──────────────────────────
            if max_hist_len:
                min_turn_i = max(0, len(l_in["history"]) - max_hist_len)
            else:
                min_turn_i = 0

            l_hist = ""
            for turn_i, (speaker, turn) in enumerate(l_in["history"]):
                if turn_i < min_turn_i:
                    continue
                assert speaker in ["user", "assistant"], f"未知发言者: {speaker}"
                l_hist += f"<{speaker}> {turn}"

            speaker, response = l_in["response"]
            assert speaker in ["user", "assistant"], f"未知发言者: {speaker}"
            l_hist_response = l_hist + "<response>" + f"<{speaker}> " + response

            # ── 实体匹配 & 序列标注 ────────────────────────
            if l_in["knowledge_base"]:
                l_hist_response, all_results = match_entities(
                    all_entities, l_hist_response, embedder, tagger, if_replace=True
                )
                hist_response_entity_ids = get_tokenized_idx(
                    l_hist_response, all_results, tripleid, tokenizer, mod
                )
            else:
                all_results = {}
                hist_response_entity_ids = {}

            l_hist_part, l_response_part = l_hist_response.split("<response>")

            hist_entity_relation_ids = delete_response_part(
                hist_response_entity_ids, l_hist_part, l_response_part, tokenizer
            )

            # ── 合并并校验长度 ────────────────────────────
            input_text += l_hist_part
            entity_relation_ids += hist_entity_relation_ids
            entity_relation_ids  = [0] + entity_relation_ids + [0]

            encoded_len = len(tokenizer.encode(input_text))
            if len(entity_relation_ids) != encoded_len:
                # eval 集：打印警告并跳过，不崩溃
                print(
                    f"  [警告] 长度不一致 idx={idx}: "
                    f"ids={len(entity_relation_ids)}, "
                    f"tokens={encoded_len}，跳过"
                )
                skipped_assert += 1
                continue

            # eval 集无 type 字段，默认为 1
            sample_type = l_in.get("type", 1)

            writer.writerow([
                entity_relation_ids,
                memory_bank,
                input_text,
                l_response_part,
                sample_type,
            ])
            written += 1

    print(f"\n[完成] 写出: {written}")
    print(f"  空KB样本（已保留）: {skipped_empty}")
    print(f"  长度不一致跳过:     {skipped_assert}")


# ─────────────────────────────────────────────
# 校验输出文件
# ─────────────────────────────────────────────
def check_entity_id(path: str):
    print(f"[校验] 检查 entity_relation_ids 与 memory_bank 一致性...")
    errors = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, row in enumerate(reader):
            entity_ids  = set(ast.literal_eval(row[0]))
            entity_ids2 = set([0])
            for mb in ast.literal_eval(row[1]):
                entity_ids2 |= set(mb)
            if entity_ids != entity_ids2:
                print(f"  [不一致] 行 {i}")
                errors += 1
    if errors == 0:
        print("  全部通过")
    else:
        print(f"  共 {errors} 处不一致")


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────
def main():
    parser = ArgumentParser(description="乌克兰数据集测试/验证集构建脚本")
    parser.add_argument("--file_entity_id",   type=str, required=True,
                        help="entity2id.txt 路径")
    parser.add_argument("--file_relation_id", type=str, required=True,
                        help="relation2id.txt 路径")
    parser.add_argument("--input_file",       type=str, required=True,
                        help="输入 txt 文件（ukraine_valid.txt 或 ukraine_test.txt）")
    parser.add_argument("--out_file",         type=str, required=True,
                        help="输出 CSV 文件路径")
    parser.add_argument("--bart_model",       type=str,
                        default="/root/autodl-fs/paper/RHO-main/bart_base",
                        help="BART 模型路径（本地路径或 HuggingFace 模型名）")
    parser.add_argument("--sbert_model",      type=str,
                        default="/root/autodl-fs/paper/RHO-main/paraphrase-distilroberta-base-v2",
                        help="SentenceTransformer 模型路径")
    parser.add_argument("--mod",              type=str, default="all",
                        choices=["all", "first"],
                        help="实体 ID 标注模式（默认 all）")
    parser.add_argument("--max_hist_len",     type=int, default=3,
                        help="最大对话历史轮数（默认 3）")
    parser.add_argument("--no_check",         action="store_true",
                        help="跳过输出文件校验")
    args = parser.parse_args()

    # 加载模型
    print(f"[模型] 加载 BART tokenizer: {args.bart_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bart_model)
    tokenizer.add_tokens(["<sep>", "<triple>", "<user>", "<assistant>", "<response>"])

    print(f"[模型] 加载 SentenceTransformer: {args.sbert_model}")
    embedder = SentenceTransformer(args.sbert_model)

    print("[模型] 加载 Flair NER tagger...")
    tagger = SequenceTagger.load("ner")

    print(f"[词典] entity2id:   {args.file_entity_id}")
    print(f"[词典] relation2id: {args.file_relation_id}")
    tripleid = Triple_id(args.file_entity_id, args.file_relation_id)

    build_eval_dataset(
        tripleid     = tripleid,
        in_path      = args.input_file,
        out_path     = args.out_file,
        tokenizer    = tokenizer,
        embedder     = embedder,
        tagger       = tagger,
        process_rel  = "process_rel",
        mod          = args.mod,
        max_hist_len = args.max_hist_len,
    )

    if not args.no_check:
        check_entity_id(args.out_file)


if __name__ == "__main__":
    main()
