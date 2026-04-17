"""
混合嵌入融合脚本
将 TransE 结构嵌入 与 Sentence-Transformer 语义嵌入 加权融合。

思路：
  - TransE 嵌入：捕捉知识图谱的拓扑结构信息（实体间的关系模式）
  - SBERT 嵌入：捕捉实体/关系名称的语义信息（即使实体在图中出现次数极少也有效）
  - 融合公式：fused = α * norm(transe) + (1-α) * norm(sbert)

用法示例：
  python fuse_embeddings.py \
      --kg_dir      ./ukraine_kg \
      --transe_dir  ./TransE_768_result \
      --output_dir  ./fused_embeddings \
      --model_name  all-mpnet-base-v2 \
      --alpha       0.3
"""

import os
import pickle
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def load_id2name(id_file: str) -> dict:
    """读取 entity2id.txt / relation2id.txt，返回 {id: name} 字典"""
    id2name = {}
    with open(id_file, "r", encoding="utf-8") as f:
        total = int(f.readline().strip())
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                name, idx = parts[0], int(parts[1])
                id2name[idx] = name
    assert len(id2name) == total, f"id 数量不一致: 声明 {total}, 实际 {len(id2name)}"
    return id2name


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """对每行做 L2 归一化，避免两种嵌入量级差异导致融合失衡"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # 防止零向量除零
    return matrix / norms


def encode_names(model: SentenceTransformer, id2name: dict, batch_size: int = 64) -> np.ndarray:
    """按 id 顺序编码名称列表，返回 numpy 矩阵 (n, hidden_dim)"""
    names = [id2name[i] for i in range(len(id2name))]
    embeddings = model.encode(
        names,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # SBERT 内部已做 L2 归一化
        convert_to_numpy=True,
    )
    return embeddings


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TransE + SBERT 嵌入融合")
    parser.add_argument("--kg_dir",     type=str, required=True,
                        help="ukraine_kg 目录，含 entity2id.txt / relation2id.txt")
    parser.add_argument("--transe_dir", type=str, required=True,
                        help="TransE 训练输出目录，含 ent_embeddings / rel_embeddings")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="融合嵌入输出目录")
    parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2",
                        help="Sentence-Transformer 模型名称（默认 all-mpnet-base-v2，768维）")
    parser.add_argument("--alpha",      type=float, default=0.3,
                        help="TransE 权重 α，SBERT 权重为 (1-α)。"
                             "α 越小语义信息越多；小数据集建议 0.2~0.4（默认 0.3）")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="SBERT 编码 batch size（默认 64）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. 读取 id→name 映射 ──────────────────
    print("[1/5] 读取实体/关系词典...")
    ent_id2name = load_id2name(os.path.join(args.kg_dir, "entity2id.txt"))
    rel_id2name = load_id2name(os.path.join(args.kg_dir, "relation2id.txt"))
    print(f"  实体数: {len(ent_id2name)}  关系数: {len(rel_id2name)}")

    # ── 2. 加载 TransE 嵌入 ───────────────────
    print("[2/5] 加载 TransE 嵌入...")
    ent_transe = pickle.load(open(os.path.join(args.transe_dir, "ent_embeddings"), "rb"))
    rel_transe = pickle.load(open(os.path.join(args.transe_dir, "rel_embeddings"), "rb"))
    print(f"  ent_transe shape: {ent_transe.shape}")
    print(f"  rel_transe shape: {rel_transe.shape}")

    transe_dim = ent_transe.shape[1]

    # ── 3. 加载 SBERT 模型并编码 ─────────────
    print(f"[3/5] 加载 Sentence-Transformer 模型: {args.model_name}")
    sbert_model = SentenceTransformer(args.model_name)
    sbert_dim = sbert_model.get_sentence_embedding_dimension()
    print(f"  SBERT 嵌入维度: {sbert_dim}")

    if sbert_dim != transe_dim:
        print(f"  [警告] 维度不一致: TransE={transe_dim}, SBERT={sbert_dim}")
        print(f"  将对 SBERT 嵌入做线性投影至 {transe_dim} 维")

    print("  编码实体名称...")
    ent_sbert = encode_names(sbert_model, ent_id2name, args.batch_size)

    print("  编码关系名称...")
    # 关系名去掉 ~ 前缀再编码，使正反向关系有相近的语义基础
    rel_id2name_clean = {
        idx: name.lstrip("~").replace("_", " ").lower()
        for idx, name in rel_id2name.items()
    }
    rel_sbert = encode_names(sbert_model, rel_id2name_clean, args.batch_size)

    # ── 4. 维度对齐（若不一致） ───────────────
    if sbert_dim != transe_dim:
        # 用随机初始化的线性层投影（保持可复现性）
        np.random.seed(42)
        proj = np.random.randn(sbert_dim, transe_dim).astype(np.float32)
        proj /= np.sqrt(sbert_dim)
        ent_sbert = ent_sbert @ proj
        rel_sbert = rel_sbert @ proj
        print(f"  投影后 SBERT 维度: {ent_sbert.shape[1]}")

    # ── 5. L2 归一化 + 加权融合 ──────────────
    print(f"[4/5] 归一化 + 加权融合 (α={args.alpha} × TransE + {1-args.alpha} × SBERT)...")
    ent_transe_n = l2_normalize(ent_transe.astype(np.float32))
    rel_transe_n = l2_normalize(rel_transe.astype(np.float32))
    ent_sbert_n  = l2_normalize(ent_sbert.astype(np.float32))
    rel_sbert_n  = l2_normalize(rel_sbert.astype(np.float32))

    alpha = args.alpha
    ent_fused = alpha * ent_transe_n + (1 - alpha) * ent_sbert_n
    rel_fused = alpha * rel_transe_n + (1 - alpha) * rel_sbert_n

    # 最终再做一次 L2 归一化，保持与原 TransE 嵌入使用方式一致
    ent_fused = l2_normalize(ent_fused)
    rel_fused = l2_normalize(rel_fused)

    print(f"  融合后 ent_fused shape: {ent_fused.shape}")
    print(f"  融合后 rel_fused shape: {rel_fused.shape}")

    # ── 6. 保存 ──────────────────────────────
    print("[5/5] 保存融合嵌入...")
    pickle.dump(ent_fused, open(os.path.join(args.output_dir, "ent_embeddings"), "wb"))
    pickle.dump(rel_fused, open(os.path.join(args.output_dir, "rel_embeddings"), "wb"))

    # 同时保存各分量，方便后续调整权重
    pickle.dump(ent_transe_n, open(os.path.join(args.output_dir, "ent_transe_norm"), "wb"))
    pickle.dump(ent_sbert_n,  open(os.path.join(args.output_dir, "ent_sbert_norm"),  "wb"))
    pickle.dump(rel_transe_n, open(os.path.join(args.output_dir, "rel_transe_norm"), "wb"))
    pickle.dump(rel_sbert_n,  open(os.path.join(args.output_dir, "rel_sbert_norm"),  "wb"))

    print(f"\n[完成] 融合嵌入已保存至: {os.path.abspath(args.output_dir)}")
    print("  ent_embeddings   — 融合实体嵌入 (供下游任务直接使用)")
    print("  rel_embeddings   — 融合关系嵌入")
    print("  ent_transe_norm  — TransE 实体嵌入（归一化后，备用）")
    print("  ent_sbert_norm   — SBERT 实体嵌入（归一化后，备用）")
    print("  rel_transe_norm  — TransE 关系嵌入（归一化后，备用）")
    print("  rel_sbert_norm   — SBERT 关系嵌入（归一化后，备用）")
    print(f"\n  模型: {args.model_name}  α(TransE)={alpha}  dim={ent_fused.shape[1]}")


if __name__ == "__main__":
    main()
