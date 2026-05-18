"""
响应构建模块

将 BART 生成的文本封装为与 page3_pipeline.Page3PipelineService.run_prediction()
完全相同的返回数据结构。

关键差异（对比 page3_pipeline 的 fallback 模式）：
- FINAL_ASSESSMENT.text  → BART 生成文本
- DIALOG[7].text         → BART 生成文本（assess 角色）
- meta.source            → "bart-model"
- 其余字段               → 与 _build_fallback() 逻辑完全相同
"""

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ─── 工具函数 ──────────────────────────────────────────────────────────

def _safe_weight(value: object) -> float:
    """与 page3_pipeline._safe_weight() 完全一致。"""
    try:
        weight = float(value)
    except (TypeError, ValueError):
        weight = 0.8
    return max(0.0, min(1.0, weight))


def _stable_vector(seed: str, dim: int) -> list:
    """与 page3_pipeline._stable_vector() 完全一致的确定性向量生成。"""
    out: list = []
    for i in range(dim):
        digest = hashlib.sha256(f"{seed}|{i}".encode("utf-8")).digest()
        frac = int.from_bytes(digest[:4], "big") / 4294967295
        out.append(round(frac * 2 - 1, 4))
    return out


# ─── 图结构转换 ────────────────────────────────────────────────────────

def triples_to_graph(triples: list) -> tuple:
    """
    将 [[head, rel, tail], ...] 转换为 page3_pipeline 使用的 nodes/links 格式。

    Returns
    -------
    (final_nodes, final_links)
        final_nodes: [{id, label, type}, ...]
        final_links: [{s, t, lb, w}, ...]
    """
    node_map: dict = {}  # label -> id (str)
    for triple in triples:
        if len(triple) != 3:
            continue
        h, r, t = triple
        if h and h not in node_map:
            node_map[h] = str(len(node_map))
        if t and t not in node_map:
            node_map[t] = str(len(node_map))

    final_nodes = [
        {"id": nid, "label": label, "type": "entity"}
        for label, nid in node_map.items()
    ]
    final_links = []
    for triple in triples:
        if len(triple) != 3:
            continue
        h, r, t = triple
        if h in node_map and t in node_map:
            final_links.append({"s": node_map[h], "t": node_map[t], "lb": r, "w": 0.85})

    return final_nodes, final_links


# ─── PROMPT_PREFIX 构建 ────────────────────────────────────────────────

def _build_prompt_prefix(
    query: str,
    final_nodes: list,
    final_links: list,
) -> dict:
    """与 page3_pipeline._build_prompt_prefix_paper_template() 完全一致。"""
    id_to_label = {
        str(n.get("id", "")): str(n.get("label", n.get("id", "?")))
        for n in final_nodes
    }
    links_sorted = sorted(
        final_links,
        key=lambda x: _safe_weight(x.get("w")),
        reverse=True,
    )
    triples_text = []
    for l in links_sorted[:8]:
        s = id_to_label.get(str(l.get("s", "")), str(l.get("s", "?")))
        t = id_to_label.get(str(l.get("t", "")), str(l.get("t", "?")))
        r = str(l.get("lb", "RELATED"))
        triples_text.append(f"[HEAD]{s} <sep> [REL]{r} <sep> [TAIL]{t}")

    if not triples_text:
        triples_text.append("[HEAD]Primary Entity <sep> [REL]RELATED <sep> [TAIL]Target Entity")

    knowledge_linear = " <triple> ".join(triples_text)
    text = f"Given the Knowledge: {knowledge_linear} <user> {query} <assistant>"
    return {"text": text, "source": "paper-3.2-template"}


# ─── TRANSE_EMBEDDING 构建 ─────────────────────────────────────────────

def _build_transe_embedding(
    final_nodes: list,
    final_links: list,
) -> dict:
    """与 page3_pipeline._build_transe_embedding_fallback() 完全一致。"""
    dim = 16
    id_to_label = {
        str(n.get("id", "")): str(n.get("label", n.get("id", "?")))
        for n in final_nodes
    }

    entity_labels = [
        str(n.get("label", n.get("id", "Entity"))).strip()
        for n in final_nodes
        if str(n.get("label", n.get("id", ""))).strip()
    ][:6]
    if not entity_labels:
        entity_labels = ["Primary Entity", "Target Entity", "Associated Entity"]

    relation_labels: list = []
    for l in final_links:
        lb = str(l.get("lb", "")).strip()
        if lb and lb not in relation_labels:
            relation_labels.append(lb)
        if len(relation_labels) >= 4:
            break
    if not relation_labels:
        relation_labels = ["RELATED", "THREATENED"]

    items: list = []
    for name in entity_labels[:6]:
        items.append({
            "name": name,
            "kind": "entity",
            "vector": _stable_vector(f"entity::{name}", dim),
        })
    for name in relation_labels[:4]:
        items.append({
            "name": name,
            "kind": "relation",
            "vector": _stable_vector(f"relation::{name}", dim),
        })

    links_sorted = sorted(
        final_links,
        key=lambda x: _safe_weight(x.get("w")),
        reverse=True,
    )
    triples_out: list = []
    for l in links_sorted[:5]:
        s = id_to_label.get(str(l.get("s", "")), str(l.get("s", "?")))
        t = id_to_label.get(str(l.get("t", "")), str(l.get("t", "?")))
        lb = str(l.get("lb", "RELATED"))
        w = _safe_weight(l.get("w"))
        score = round(max(0.55, min(0.98, 0.58 + w * 0.4)), 3)
        triples_out.append({"triple": f"{s}-[{lb}]->{t}", "score": score})

    if not triples_out:
        triples_out = [
            {"triple": "Primary Entity-[RELATED]->Target Entity", "score": 0.73},
            {"triple": "Target Entity-[THREATENED]->Associated Entity", "score": 0.69},
        ]

    return {
        "dim": dim,
        "tensor_shape": [len(items), dim],
        "items": items,
        "triples": triples_out,
        "note": (
            "Vectors are rule-based + LLM-simulated, for UI demonstration only, "
            "not representative of real training parameters."
        ),
        "source": "fallback-simulated",
    }


# ─── 主构建函数 ────────────────────────────────────────────────────────

def build_full_response(
    query: str,
    triples: list,
    history: str,
    bart_response: str,
) -> dict:
    """
    将 BART 生成文本封装为与 run_prediction() 完全相同的返回结构。

    Parameters
    ----------
    query : str
        用户查询字符串，用于 DIALOG[0] 和结构化字段。
    triples : list[list[str, str, str]]
        子图三元组。
    history : str
        已格式化的对话上下文文本（仅用于记录，不影响结构构建）。
    bart_response : str
        BART 模型生成的回复文本，将填入 FINAL_ASSESSMENT 和 DIALOG[7]。

    Returns
    -------
    dict
        包含 DIALOG / METRICS / REASONING_FRAMEWORK / FINAL_ASSESSMENT /
        STRATEGIES / PROMPT_PREFIX / TRANSE_EMBEDDING / meta 的完整响应。
    """
    # 将三元组转换为 nodes/links 格式（供后续构建函数使用）
    final_nodes, final_links = triples_to_graph(triples)

    # ── 提取关键图元素（与 page3_pipeline._build_fallback() 一致）──
    labels = [str(n.get("label", "Entity")) for n in final_nodes[:6]]
    e0 = labels[0] if len(labels) > 0 else "Primary Entity"
    e1 = labels[1] if len(labels) > 1 else "Target Entity"
    e2 = labels[2] if len(labels) > 2 else "Associated Entity"
    rel0 = str(final_links[0].get("lb", "RELATED")) if final_links else "RELATED"
    rel1 = str(final_links[1].get("lb", "DEPLOYED")) if len(final_links) > 1 else "DEPLOYED"
    w0 = final_links[0].get("w", 0.85) if final_links else 0.85
    w1 = final_links[1].get("w", 0.85) if len(final_links) > 1 else 0.85

    # ── DIALOG（8 条，与 _build_fallback() 完全一致，仅 [7] 使用 bart_response）──
    dialog: list = [
        {
            "role": "user",
            "text": f"[OSINT Query — Behavior Reasoning] {query}",
        },
        {
            "role": "action",
            "text": f"kgwalk/choose_path → [{e0}, {rel0}, {e1}]",
            "dim": 1,
        },
        {
            "role": "asst",
            "text": (
                f"Observable facts confirm that {e0} has established a documented operational pattern "
                f"against {e1} via the {rel0} relationship chain (w={w0}). "
                "This behavior is consistently recorded across multiple subgraph paths, "
                "indicating a systematic and deliberate course of action."
            ),
            "dim": 1,
        },
        {
            "role": "action",
            "text": f"kgwalk/choose_path → [{e1}, {rel1}, {e2}]",
            "dim": 2,
        },
        {
            "role": "asst",
            "text": (
                f"Underlying motives of {e0} can be inferred from the documented facts. "
                f"The {rel1} relationship between {e1} and {e2} (w={w1}) reveals a concealed strategic driver "
                "that goes beyond the surface-level observable actions. "
                "Available intelligence supports an escalatory intent already embedded in the existing subgraph data."
            ),
            "dim": 2,
        },
        {
            "role": "action",
            "text": f"kgwalk/choose_path → [{e0}, {rel0}, {e1}]",
            "dim": 3,
        },
        {
            "role": "asst",
            "text": (
                f"Based on observable actions and inferred motives, the latent behavior of {e0} "
                f"with respect to {e1} is assessed to already be underway. "
                "High-confidence subgraph indicators (w>0.75) consistently converge on this inference."
            ),
            "dim": 3,
        },
        {
            "role": "assess",
            # ← BART 生成文本替换此处
            "text": bart_response if bart_response else (
                f"Assessment: Based on available indicators from the knowledge graph subgraph, "
                f"{e0} is assessed to be engaged in latent escalatory behavior targeting {e1}. "
                f"The {rel0} relationship chain presents high confidence (w={w0}) as the primary behavioral indicator. "
                f"Converging evidence from {rel1} (w={w1}) further corroborates this inference. "
                "Recommend elevated monitoring posture across all identified subgraph entities."
            ),
        },
    ]

    # ── METRICS（与 _build_fallback() 完全一致）──
    metrics: dict = {
        "threat_level": "High",
        "probability":  75,
        "confidence":   82,
    }

    # ── REASONING_FRAMEWORK（与 _build_fallback() 完全一致）──
    reasoning_framework: list = [
        {"step": 1, "text": f"Observable actions: {e0} has documented a {rel0} action pattern against {e1}, subgraph path confirms w={w0}"},
        {"step": 2, "text": f"Underlying motives: Based on {rel0}/{rel1} relation chains, concealed strategic intent points toward sustained pressure on {e1}"},
        {"step": 3, "text": f"Inferred latent behavior: {e0} is assessed to be already engaged in covert {rel0}-related operations against {e1}"},
    ]

    # ── FINAL_ASSESSMENT（← BART 生成文本）──
    final_assessment: dict = {
        "text": bart_response if bart_response else (
            f"Assessment: Based on available indicators from the knowledge graph subgraph, "
            f"{e0} is assessed to be engaged in latent activities involving {e1}. "
            f"The {rel0} relationship chain demonstrates high confidence (w={w0}) as the primary behavioral indicator. "
            f"Secondary evidence from the {rel1} path (w={w1}) provides additional corroboration. "
            "With moderate-to-high confidence, the latent behavior is assessed to already be underway. "
            "Recommend continued monitoring of all identified entities and relationship chains."
        )
    }

    # ── STRATEGIES（与 _build_fallback() 完全一致）──
    strategies: list = [
        {
            "id": "A",
            "title": f"🎯 推理方案 A：{e0} 主动实施 {rel0} 潜藏行动",
            "probability_text": "置信度 65%",
            "desc": (
                f"基于{e0}针对{e1}的{rel0}行动轨迹，推断其正在实施更大规模的隐蔽行动，"
                f"利用{rel0}关系链进行战略部署以实现既定战略目标，"
                f"对目标施加持续压力。子图置信权重 w={w0} 为此推理提供高置信度支撑。"
            ),
            "evidence": [
                {"text": f"{e0} →[{rel0}]→ {e1} (w={w0}) → 已记录行为轨迹，高置信度确认"},
                {"text": f"{e1} →[{rel1}]→ {e2} (w={w1}) → 战略关联节点激活确认"},
            ],
            "pros": [
                f"子图证据链完整，{rel0}关系路径置信度高，推理依据充分",
                "多条子图路径收敛，推理结论一致性强",
            ],
            "cons": [
                "潜藏行为难以直接观测，存在替代解读的可能性",
                "子图数据覆盖范围有限，可能遗漏部分关键证据",
            ],
        },
        {
            "id": "B",
            "title": f"⚡ 推理方案 B：{e0} 转向防御性战略调整",
            "probability_text": "置信度 25%",
            "desc": (
                f"{e0}的潜藏行为也可能是防御性重新部署，沿{rel1}路径向{e2}转移主力，"
                f"加强防御纵深以应对潜在反制威胁，保存核心战略资产。"
                f"此推理置信度较低（25%），主要基于{rel1}（w={w1}）的防御信号。"
            ),
            "evidence": [
                {"text": f"{e1} →[{rel1}]→ {e2} (w={w1}) → 防御部署信号确认"},
            ],
            "pros": [
                f"保存{e0}核心战略资产，降低直接对抗风险",
                "减少过度升级带来的国际压力",
            ],
            "cons": [
                "被视为战略退缩信号，削弱自身战略主动性",
                "防御姿态可能鼓励对方进一步施压",
            ],
        },
    ]

    # ── PROMPT_PREFIX（与 _build_prompt_prefix_paper_template() 完全一致）──
    prompt_prefix = _build_prompt_prefix(query, final_nodes, final_links)

    # ── TRANSE_EMBEDDING（与 _build_transe_embedding_fallback() 完全一致）──
    transe_embedding = _build_transe_embedding(final_nodes, final_links)

    # ── meta（标识数据来源）──
    meta: dict = {
        "source":        "bart-model",
        "use_llm":       False,
        "llm_attempted": False,
        "llm_succeeded": False,
        "fallback_used": False,
        "llm_error":     None,
    }

    return {
        "DIALOG":              dialog,
        "METRICS":             metrics,
        "REASONING_FRAMEWORK": reasoning_framework,
        "FINAL_ASSESSMENT":    final_assessment,
        "STRATEGIES":          strategies,
        "PROMPT_PREFIX":       prompt_prefix,
        "TRANSE_EMBEDDING":    transe_embedding,
        "meta":                meta,
    }
