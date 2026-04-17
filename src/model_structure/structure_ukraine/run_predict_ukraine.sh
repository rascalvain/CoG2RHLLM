#!/bin/bash
# ============================================================
# Ukraine 数据集 RHO 模型 —— 测试集推理脚本
# 使用 structure_ukraine 版本代码
# ============================================================

# ──────────────────── 路径配置（与训练脚本保持一致）────────────────────
# 训练好的模型检查点目录（即训练脚本中的 OUTDIR）
MODEL=/home/shu1004/lyx/RHO-main/src/structure_file/structure_ukraine/ukraine_output_contrast_geometric_tau5e-1

# Ukraine 数据集根目录
DATA=/home/shu1004/lyx/RHO-main/ukrine/process_data

# 融合嵌入路径（与训练时保持一致）
EMBEDPATH=${DATA}/fused_embeddings/ent_rel_embeddings
# 若未融合，改为 TransE 原始嵌入：
# EMBEDPATH=${DATA}/train_transe/TransE_768_result/ent_rel_embeddings

# 脚本所在目录
SCRIPT_DIR=/home/shu1004/lyx/RHO-main/src/structure_file/structure_ukraine

# ──────────────────── 输出配置 ────────────────────
# 生成文本保存文件名（最终文件路径为 MODEL/GENERATION_FILE）
GENERATION_FILE="B4_ukraine_test_predictions.txt"

# 日志目录
LOG_DIR=${SCRIPT_DIR}/logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_DIR}/predict_ukraine_${TIMESTAMP}.log

mkdir -p "${LOG_DIR}"

echo "========================================"
echo "  Ukraine RHO 测试集推理启动"
echo "  模型路径:    ${MODEL}"
echo "  测试数据:    ${DATA}/ukraine_test_dataset.csv"
echo "  生成结果:    ${MODEL}/${GENERATION_FILE}"
echo "  日志文件:    ${LOG_FILE}"
echo "========================================"

nohup bash -c "
CUDA_VISIBLE_DEVICES=0 \
python ${SCRIPT_DIR}/run_summarization_attention_geometric_gate.py \
  --use_memory_bank=True \
  --memory_bank_mode='all' \
  --use_kg_embedding=True \
  --mode=input \
  --model_name_or_path=${MODEL} \
  --cache_dir=${MODEL}/cache_dir \
  --text_column=history \
  --summary_column=response \
  --test_file=${DATA}/ukraine_test_dataset.csv \
  --train_file=${DATA}/ukraine_train_dataset.csv \
  --entity_relation_embedding_path=${EMBEDPATH} \
  --pad_to_max_length=True \
  --output_dir=${MODEL} \
  --do_predict=True \
  --predict_with_generate=True \
  --num_beams=4 \
  --max_source_length=800 \
  --max_target_length=64 \
  --per_device_eval_batch_size=4 \
  --generation_file=${GENERATION_FILE} \
  --overwrite_cache \
  --report_to='none'
" > "${LOG_FILE}" 2>&1 &

echo "后台进程 PID: $!"
echo "查看日志:  tail -f ${LOG_FILE}"
echo "生成结果:  ${MODEL}/${GENERATION_FILE}"
