"""
API 配置中心

所有路径和运行参数在此处统一管理，其他模块只从此处导入。
部署时按实际环境填写 MODEL_CHECKPOINT_DIR 和 KG_EMBEDDING_PATH。
"""

import os

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── 项目根目录 ────────────────────────────────────────────────────────
# api/ 的上级即为项目根（G:\小论文\代码整理\第二章）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 模型路径（部署时填写）────────────────────────────────────────────
# 训练好的模型 checkpoint 目录（含 pytorch_model.bin / config.json 等）
MODEL_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "")  # TODO: 填写实际路径

# KG 嵌入 pkl 文件路径（无扩展名，与训练脚本 EMBEDPATH 变量保持一致）
# 例：os.path.join(PROJECT_ROOT, "russian", "process_data", "fused_embeddings", "ent_rel_embeddings")
KG_EMBEDDING_PATH = os.path.join(PROJECT_ROOT, "")     # TODO: 填写实际路径

# ── KG 字典路径（通常无需修改）───────────────────────────────────────
ENTITY2ID_PATH = os.path.join(
    PROJECT_ROOT, "russian", "process_data", "ukraine_kg", "entity2id.txt"
)
RELATION2ID_PATH = os.path.join(
    PROJECT_ROOT, "russian", "process_data", "ukraine_kg", "relation2id.txt"
)

# ── BartConfig 注入参数（与 run_summarization_attention_geometric_gate.py L435-449 保持一致）──
MODEL_CONFIG = {
    "mode":                       "input",
    "memory_bank_mode":           "all",
    "use_memory_bank":            True,
    "use_kg_embedding":           True,
    "prefix_tuning":              False,
    "freeze_model":               True,
    "pre_seq_len":                4,
    "prefix_projection":          False,
    "prefix_hidden_size":         512,
    "prefix_hidden_dropout_prob": 0.1,
}

# ── 生成超参（与推理脚本保持一致）────────────────────────────────────
GENERATE_KWARGS: dict = {
    "num_beams":  4,
    "max_length": 64,
}

# ── Tokenizer 特殊 token（与训练脚本 L463 完全一致）──────────────────
SPECIAL_TOKENS: list = ["<sep>", "<triple>", "<user>", "<assistant>", "<response>"]

# ── 数据处理参数 ─────────────────────────────────────────────────────
MAX_SOURCE_LENGTH: int = 800   # 与训练脚本 max_source_length 一致
ENTITY_ID_MOD: str     = "all" # token 标注模式：all=每个 subword 都标注

# ── 推理设备 ────────────────────────────────────────────────────────
if _TORCH_AVAILABLE:
    import torch
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cpu"

# ── Flask 服务参数 ────────────────────────────────────────────────────
FLASK_HOST:  str  = "0.0.0.0"
FLASK_PORT:  int  = 5000
FLASK_DEBUG: bool = False   # 生产环境设为 False，避免 reloader 导致模型二次加载
