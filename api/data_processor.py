"""
数据预处理模块

将 API 接收的 (history, triples) 转换为模型所需的张量输入。
严格复现 build_dataset_ukraine.py 中 entity_relation_ids 的构建逻辑。
"""

import logging
import re
import unicodedata

import config

logger = logging.getLogger(__name__)


# ─── 工具函数 ──────────────────────────────────────────────────────────

def to_ascii(text: str) -> str:
    """与 build_dataset_ukraine.py:L40-44 完全一致的 ASCII 规范化。"""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", errors="ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip()


# ─── DataProcessor ─────────────────────────────────────────────────────

class DataProcessor:
    """将 history + triples 构建为 BART 模型推理所需的全部输入。"""

    def __init__(self, tripleid, tokenizer, mod: str = "all"):
        """
        Parameters
        ----------
        tripleid : Triple_id
            已加载 entity2id / relation2id 的 Triple_id 实例。
        tokenizer : PreTrainedTokenizer
            已添加特殊 token 的 BART tokenizer。
        mod : str
            实体标注模式，"all" 表示每个 subword 都标注（与训练一致）。
        """
        self.tripleid = tripleid
        self.tokenizer = tokenizer
        self.mod = mod

    # ─────────────────────────── 主方法 ───────────────────────────

    def build_inputs(self, history: str, triples: list) -> dict:
        """
        将 history + triples 构建为模型 generate() 所需的输入字典。

        Parameters
        ----------
        history : str
            对话历史文本（已格式化，含 <user>/<assistant> 等特殊 token）。
        triples : list[list[str, str, str]]
            子图三元组，每个元素为 [head, relation, tail]。

        Returns
        -------
        dict
            {input_ids, input_entity_relation_ids, memory_bank,
             memory_bank_attention_mask, attention_mask}
            值均为 Python list（非张量），由 inference.py 负责转张量。
        """
        tokenizer = self.tokenizer
        tripleid = self.tripleid

        # ── 步骤 1: 知识前缀 entity_relation_ids ──
        # 对应 build_dataset_ukraine.py:L88-192
        pre = "Given the knowledge:"
        pre_len = len(tokenizer.tokenize(pre))
        entity_relation_ids = [0] * pre_len
        memory_bank = []
        triples_text_parts = []

        for triple in triples:
            if len(triple) != 3:
                logger.warning("三元组格式错误（长度≠3），跳过: %s", triple)
                continue

            sub_raw, rel_raw, obj_raw = triple
            sub = to_ascii(sub_raw)
            rel = to_ascii(rel_raw)
            obj = to_ascii(obj_raw)

            if not sub or not rel or not obj:
                logger.warning("ASCII 规范化后为空，跳过: %s", triple)
                continue

            # 反向关系处理（与 build_dataset_ukraine.py:L137-139 一致）
            if "~" in rel:
                rel = rel.replace("~", "")
                sub, obj = obj, sub

            # 查字典获取 ID
            try:
                sub_id = tripleid.get_triple_id(sub, relation=False)
            except KeyError:
                logger.warning("实体不在词典中: '%s'，跳过三元组 %s", sub, triple)
                continue
            try:
                rel_id = tripleid.get_triple_id(rel, relation=True)
            except KeyError:
                logger.warning("关系不在词典中: '%s'，跳过三元组 %s", rel, triple)
                continue
            try:
                obj_id = tripleid.get_triple_id(obj, relation=False)
            except KeyError:
                logger.warning("实体不在词典中: '%s'，跳过三元组 %s", obj, triple)
                continue

            # 关系名文本化（tokenize 时使用空格分隔词，与训练一致）
            rel_text = re.sub(r"[-_]", " ", rel)

            # 计算各部分 subword token 数（注意前缀空格，与训练一致）
            sub_len = len(tokenizer.tokenize(" " + sub))
            rel_len = len(tokenizer.tokenize(" " + rel_text))
            obj_len = len(tokenizer.tokenize(" " + obj))

            memory_bank.append([sub_id, rel_id, obj_id])

            # entity_relation_ids 拼装
            # 对应 build_dataset_ukraine.py:L164-168
            # [sub_id]*sub_len + [0]    → sub tokens + <sep> token
            # [rel_id]*rel_len + [0]    → rel tokens + <sep> token
            # [obj_id]*obj_len + [0]    → obj tokens + <triple> token（末尾多余会被移除）
            if self.mod == "all":
                entity_relation_ids += (
                    [sub_id] * sub_len + [0]
                    + [rel_id] * rel_len + [0]
                    + [obj_id] * obj_len + [0]
                )
            elif self.mod == "first":
                entity_relation_ids += (
                    [sub_id] + [0] * sub_len
                    + [rel_id] + [0] * rel_len
                    + [obj_id] + [0] * obj_len
                )
            else:
                raise ValueError(f"不支持的 mod: {self.mod}")

            # 文本拼装（与 build_dataset_ukraine.py:L187 一致）
            triples_text_parts.append(
                f"{sub}<sep> {rel_text}<sep> {obj}"
            )

        # 移除末尾多余的 [0]（对应 L191）
        if triples_text_parts:
            entity_relation_ids = entity_relation_ids[:-1]

        # 拼装知识前缀文本（对应 L192）
        if triples_text_parts:
            knowledge_text = pre + " " + "<triple> ".join(triples_text_parts)
        else:
            knowledge_text = pre

        # ── 步骤 2: 对话历史 entity_relation_ids（简化：全零）──
        # 训练时使用 SentenceTransformer + Flair NER，API 中简化为全零
        history_tokens = tokenizer.tokenize(history)
        history_entity_ids = [0] * len(history_tokens)

        # ── 步骤 3: 合并 + BOS/EOS 对齐 ──
        # 对应 build_dataset_ukraine.py:L235-237
        full_text = knowledge_text + history
        entity_relation_ids_full = (
            [0]                       # BOS token <s>
            + entity_relation_ids     # 知识前缀部分
            + history_entity_ids      # 对话历史部分
            + [0]                     # EOS token </s>
        )

        encoded = tokenizer.encode(full_text)  # 含 BOS/EOS
        if len(entity_relation_ids_full) != len(encoded):
            logger.error(
                "entity_relation_ids 长度(%d) 与 tokenizer.encode 长度(%d) 不一致，"
                "full_text='%s...'",
                len(entity_relation_ids_full),
                len(encoded),
                full_text[:100],
            )
            # 强制对齐：截断或补零
            if len(entity_relation_ids_full) > len(encoded):
                entity_relation_ids_full = entity_relation_ids_full[: len(encoded)]
            else:
                entity_relation_ids_full += [0] * (
                    len(encoded) - len(entity_relation_ids_full)
                )

        # ── 步骤 4: 截断到最大输入长度 ──
        max_len = config.MAX_SOURCE_LENGTH
        input_ids = encoded[:max_len]
        entity_ids_truncated = entity_relation_ids_full[:max_len]
        attention_mask = [1] * len(input_ids)

        # memory_bank 空保护（至少 1 行，mask=0 让模型忽略）
        if not memory_bank:
            memory_bank = [[0, 0, 0]]
            memory_bank_attention_mask = [0]
        else:
            memory_bank_attention_mask = [1] * len(memory_bank)

        return {
            "input_ids":                  input_ids,
            "input_entity_relation_ids":  entity_ids_truncated,
            "memory_bank":                memory_bank,
            "memory_bank_attention_mask": memory_bank_attention_mask,
            "attention_mask":             attention_mask,
        }
