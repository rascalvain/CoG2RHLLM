"""
模型推理模块

负责一次性加载模型 / tokenizer / KG 嵌入，并提供 generate_response() 接口。
严格复现 run_summarization_attention_geometric_gate.py:L404-490 的加载流程。
"""

import logging
import os
import pickle
import sys

import numpy as np
import torch

import config
from data_processor import DataProcessor

logger = logging.getLogger(__name__)

# ── 模块级延迟变量（由 load_model 初始化）──────────────────────────────
_model = None
_tokenizer = None
_tripleid = None
_processor = None
_device = None


def load_model():
    """
    一次性加载所有推理所需资源。在 Flask 启动时调用一次。

    加载流程严格复现 run_summarization_attention_geometric_gate.py:L404-490：
    1. KG 嵌入 → 2. BartConfig → 3. Tokenizer → 4. 模型 → 5. Triple_id → 6. DataProcessor
    """
    global _model, _tokenizer, _tripleid, _processor, _device

    _device = torch.device(config.DEVICE)
    logger.info("推理设备: %s", _device)

    # ── 1. 加载 KG 嵌入（L404-406）──────────────────────────────────
    logger.info("加载 KG 嵌入: %s", config.KG_EMBEDDING_PATH)
    with open(config.KG_EMBEDDING_PATH, "rb") as f:
        kg_emb = np.array(pickle.load(f))
    original_dim = kg_emb.shape[1]
    # 首位插入全零行（padding ID=0 对应零向量）
    kg_emb = np.array(
        list(np.zeros((1, original_dim))) + list(kg_emb)
    )
    logger.info("KG 嵌入维度: %s (含 padding 行)", kg_emb.shape)

    # ── 2. 加载 BartConfig 并注入自定义参数（L427-449）────────────────
    from transformers import AutoConfig
    bart_config = AutoConfig.from_pretrained(config.MODEL_CHECKPOINT_DIR)
    bart_config.update(config.MODEL_CONFIG)
    logger.info("BartConfig 已加载并注入自定义参数")

    # ── 3. 加载 Tokenizer 并添加特殊 token（L454-463）────────────────
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT_DIR)
    _tokenizer.add_tokens(config.SPECIAL_TOKENS)
    logger.info("Tokenizer 已加载, vocab_size=%d", len(_tokenizer))

    # ── 4. 加载模型（L478-490）───────────────────────────────────────
    # 将 structure_ukraine 目录加入 sys.path 以便 import 自定义模型类
    model_src_dir = os.path.join(
        config.PROJECT_ROOT, "src", "model_structure", "structure_ukraine"
    )
    if model_src_dir not in sys.path:
        sys.path.insert(0, model_src_dir)

    from modeling_bart_contrast_attention_geometric_gate import (
        BartForConditionalGeneration,
    )

    logger.info("加载模型: %s", config.MODEL_CHECKPOINT_DIR)
    _model = BartForConditionalGeneration.from_pretrained(
        config.MODEL_CHECKPOINT_DIR,
        config=bart_config,
        entity_relation_weight=kg_emb,  # 通过 **kwargs 传入 __init__
    )
    _model.resize_token_embeddings(len(_tokenizer))
    _model.eval()
    _model.to(_device)
    logger.info("模型已加载并移至 %s", _device)

    # ── 5. 加载 Triple_id（引用 src/data/data_utils.py）──────────────
    data_src_dir = os.path.join(config.PROJECT_ROOT, "src", "data")
    if data_src_dir not in sys.path:
        sys.path.insert(0, data_src_dir)

    from data_utils import Triple_id

    _tripleid = Triple_id(config.ENTITY2ID_PATH, config.RELATION2ID_PATH)
    logger.info(
        "Triple_id 已加载: %d 实体, %d 关系",
        _tripleid.num_entity,
        len(_tripleid.relation_id),
    )

    # ── 6. 初始化 DataProcessor ──────────────────────────────────────
    _processor = DataProcessor(_tripleid, _tokenizer, mod=config.ENTITY_ID_MOD)
    logger.info("DataProcessor 已初始化")


def generate_response(history: str, triples: list) -> str:
    """
    执行一次 BART 推理，返回解码后的回复文本。

    Parameters
    ----------
    history : str
        已格式化的对话上下文文本。
    triples : list[list[str, str, str]]
        子图三元组列表。

    Returns
    -------
    str
        模型生成的回复文本。
    """
    if _model is None:
        raise RuntimeError("模型未初始化，请先调用 load_model()")

    # 1. 数据预处理
    inputs = _processor.build_inputs(history, triples)

    # 2. 转为张量，batch_size=1
    input_ids = torch.LongTensor([inputs["input_ids"]]).to(_device)
    attention_mask = torch.LongTensor([inputs["attention_mask"]]).to(_device)
    input_entity_relation_ids = torch.LongTensor(
        [inputs["input_entity_relation_ids"]]
    ).to(_device)
    memory_bank = torch.LongTensor([inputs["memory_bank"]]).to(_device)
    memory_bank_attention_mask = torch.LongTensor(
        [inputs["memory_bank_attention_mask"]]
    ).to(_device)

    # 3. 推理
    with torch.no_grad():
        output_ids = _model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_entity_relation_ids=input_entity_relation_ids,
            memory_bank=memory_bank,
            memory_bank_attention_mask=memory_bank_attention_mask,
            **config.GENERATE_KWARGS,
        )

    # 4. 解码
    response = _tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return response.strip()
