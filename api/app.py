"""
Flask API 入口

提供两个端点：
  POST /predict  — 接收 query / history / triples，返回与 page3_pipeline 对齐的完整结构化响应
  GET  /health   — 健康检查

启动方式：
  cd api
  python app.py
"""

import logging
import os
import sys

# 确保 api/ 目录在路径中（支持从任意工作目录启动）
_API_DIR = os.path.dirname(os.path.abspath(__file__))
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from flask import Flask, jsonify, request

import config
import inference
import response_builder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ─── 路由 ──────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    主推理端点。

    请求体（JSON）：
    {
        "query":   "目标行为推理查询字符串",
        "history": "Given the knowledge: ...<user> ...<assistant>",
        "triples": [["head", "relation", "tail"], ...]
    }

    返回（JSON）：与 page3_pipeline.run_prediction() 完全相同的结构。
    """
    # ── 1. 解析请求 ──────────────────────────────────────────────────
    if not request.is_json:
        return jsonify({"error": "请求必须为 JSON 格式 (Content-Type: application/json)"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "无法解析 JSON 请求体"}), 400

    # ── 2. 字段校验 ──────────────────────────────────────────────────
    history = data.get("history")
    if not isinstance(history, str) or not history.strip():
        return jsonify({"error": "'history' 字段必须为非空字符串"}), 400

    triples = data.get("triples")
    if not isinstance(triples, list):
        return jsonify({"error": "'triples' 字段必须为列表"}), 400

    for i, triple in enumerate(triples):
        if not isinstance(triple, list) or len(triple) != 3:
            return jsonify({
                "error": f"triples[{i}] 格式错误，每个三元组必须是长度为 3 的列表 [head, relation, tail]"
            }), 400
        if not all(isinstance(x, str) for x in triple):
            return jsonify({
                "error": f"triples[{i}] 中的所有元素必须为字符串"
            }), 400

    query = data.get("query", "")
    if not isinstance(query, str):
        query = str(query)

    # ── 3. BART 推理 ──────────────────────────────────────────────────
    try:
        bart_response = inference.generate_response(history, triples)
    except RuntimeError as e:
        logger.error("推理时发生 RuntimeError: %s", e, exc_info=True)
        return jsonify({"error": f"服务内部错误（模型未初始化？）: {e}"}), 500
    except Exception as e:  # noqa: BLE001
        logger.error("推理时发生未预期错误: %s", e, exc_info=True)
        return jsonify({"error": "推理失败，请检查输入或联系管理员"}), 500

    # ── 4. 构建完整结构化响应 ────────────────────────────────────────
    try:
        result = response_builder.build_full_response(
            query=query,
            triples=triples,
            history=history,
            bart_response=bart_response,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("构建响应时发生错误: %s", e, exc_info=True)
        return jsonify({"error": "响应构建失败"}), 500

    return jsonify(result), 200


@app.route("/health", methods=["GET"])
def health():
    """健康检查端点。"""
    model_loaded = inference._model is not None
    return jsonify({
        "status":       "ok" if model_loaded else "loading",
        "model_loaded": model_loaded,
    }), 200 if model_loaded else 503


# ─── 启动 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("正在加载模型，请稍候...")
    try:
        inference.load_model()
        logger.info("模型加载完毕，服务就绪")
    except Exception as e:
        logger.error("模型加载失败: %s", e, exc_info=True)
        raise

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        use_reloader=False,   # 禁止热重载，避免模型二次加载
    )
