"""
数据集构建脚本：适配乌克兰数据集
参照 src/data/build_dataset_new.py 的处理逻辑。

主要流程：
  1. 读取负采样后的训练文件（ukraine_train_with_neg.txt）
  2. 利用 entity2id.txt / relation2id.txt 将三元组转为 ID 序列
  3. 用 BART tokenizer 对"知识前缀 + 对话历史"进行分词，
     同时构建 entity_relation_ids（每个 token 对应的实体/关系 ID）
  4. 用 SentenceTransformer + Flair NER 在对话文本中定位实体提及
  5. 写出 CSV 文件供后续 BART 模型训练使用

输出 CSV 列：entity_relation_ids | memory_bank | history | response | type

用法示例：
  python build_dataset_ukraine.py \\
      --file_entity_id   ./ukraine_kg/entity2id.txt \\
      --file_relation_id ./ukraine_kg/relation2id.txt \\
      --input_file       ./ukraine_train_with_neg.txt \\
      --out_file         ./ukraine_train_dataset.csv \\
      --bart_model       /path/to/bart_base \\
      --sbert_model      /path/to/paraphrase-distilroberta-base-v2 \\
      --max_hist_len     3 \\
      --mod              all
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

# ── 定位 data_utils.py（相对路径兼容服务器环境）──────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
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
def build_dataset(
    tripleid,
    in_path: str,
    out_path: str,
    tokenizer,
    embedder,
    tagger,
    process_rel: str = "process_rel",
    mod: str = "all",
    max_hist_len: int = 3,
    skip_empty_kb: bool = False,
    filter_no_english: bool = False,
):
    pre = "Given the knowledge:"
    pre_len = len(tokenizer.tokenize(pre))

    skipped_empty  = 0
    skipped_assert = 0
    written        = 0

    with open(in_path, encoding="utf-8") as f_in, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:

        writer = csv.writer(fout)
        writer.writerow(["entity_relation_ids", "memory_bank", "history", "response", "type"])

        for idx, line in tqdm(enumerate(f_in.readlines()), desc="构建数据集"):
            if idx % 500 == 0:
                print(f"  处理第 {idx} 行...")

            l_in = json.loads(line)

            # ── 处理空 KB 样本（最终总结轮）─────────────────
            if not l_in["knowledge_base"]:
                if skip_empty_kb:
                    skipped_empty += 1
                    continue
                # 不跳过时：input_text 仅含前缀，entity_relation_ids 全零
                entity_relation_ids = [0] * pre_len
                memory_bank         = []
                input_text          = pre
                if_only_english     = True
            else:
                # ── 处理含 KB 样本 ────────────────────────
                entity_relation_ids = [0] * pre_len
                memory_bank         = []
                triples             = []
                if_only_english     = True
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

                    if filter_no_english:
                        if_only_english = (
                            if_only_english
                            and only_english(sub)
                            and only_english(rel_text)
                            and only_english(obj)
                        )

                    triple_text = sub + "<sep> " + rel_text + "<sep> " + obj
                    triples.append(triple_text)

                # 移除最后一个多余的 <triple> 位置
                entity_relation_ids = entity_relation_ids[:-1]
                input_text = pre + " " + "<triple> ".join(triples)

            if filter_no_english and not if_only_english:
                print(f"  [跳过] 非纯英文样本 idx={idx}")
                continue

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
                # 无 KB：对话历史无实体标注，全零
                all_results = {}
                hist_response_entity_ids = {}

            l_hist_part, l_response_part = l_hist_response.split("<response>")

            hist_entity_relation_ids = delete_response_part(
                hist_response_entity_ids, l_hist_part, l_response_part, tokenizer
            )

            # ── 合并并校验长度 ────────────────────────────
            input_text += l_hist_part
            entity_relation_ids += hist_entity_relation_ids
            entity_relation_ids  = [0] + entity_relation_ids + [0]  # <s> 和 </s>

            encoded_len = len(tokenizer.encode(input_text))
            if len(entity_relation_ids) != encoded_len:
                print(
                    f"  [警告] 长度不一致 idx={idx}: "
                    f"entity_relation_ids={len(entity_relation_ids)}, "
                    f"tokenizer={encoded_len}，跳过"
                )
                skipped_assert += 1
                continue

            sample_type = l_in.get("type", 1)
            writer.writerow([
                entity_relation_ids,
                memory_bank,
                input_text,
                l_response_part,
                sample_type,
            ])
            written += 1

    print(f"\n[完成] 写出: {written}  跳过(空KB): {skipped_empty}  跳过(长度异常): {skipped_assert}")


# ─────────────────────────────────────────────
# 校验输出文件（与原脚本保持一致）
# ─────────────────────────────────────────────
def check_entity_id(path: str):
    print(f"[校验] 检查 entity_relation_ids 与 memory_bank 一致性: {path}")
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
                print(f"  [不一致] 行 {i}: ids={entity_ids} vs mb={entity_ids2}")
                errors += 1
    if errors == 0:
        print("  全部通过")
    else:
        print(f"  共 {errors} 处不一致")


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────
def main():
    parser = ArgumentParser(description="乌克兰数据集构建脚本（适配 build_dataset_new.py）")
    parser.add_argument("--file_entity_id",   type=str, required=True,
                        help="entity2id.txt 路径（ukraine_kg/entity2id.txt）")
    parser.add_argument("--file_relation_id", type=str, required=True,
                        help="relation2id.txt 路径（ukraine_kg/relation2id.txt）")
    parser.add_argument("--input_file",       type=str, required=True,
                        help="输入 txt 文件（ukraine_train_with_neg.txt 等）")
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
    parser.add_argument("--skip_empty_kb",    action="store_true",
                        help="跳过 knowledge_base 为空的样本（最终总结轮）")
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

    # 加载实体/关系词典
    print(f"[词典] entity2id: {args.file_entity_id}")
    print(f"[词典] relation2id: {args.file_relation_id}")
    tripleid = Triple_id(args.file_entity_id, args.file_relation_id)

    # 构建数据集
    build_dataset(
        tripleid       = tripleid,
        in_path        = args.input_file,
        out_path       = args.out_file,
        tokenizer      = tokenizer,
        embedder       = embedder,
        tagger         = tagger,
        process_rel    = "process_rel",
        mod            = args.mod,
        max_hist_len   = args.max_hist_len,
        skip_empty_kb  = args.skip_empty_kb,
    )

    # 校验
    if not args.no_check:
        check_entity_id(args.out_file)


if __name__ == "__main__":
    main()
