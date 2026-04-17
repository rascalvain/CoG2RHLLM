#!/bin/bash
# ============================================================
# Ukraine 数据集 RHO 模型训练脚本
# 使用 structure_ukraine 版本代码（支持动态三元组数 + 空KB + 预建正负样本）
# ============================================================

# ──────────────────── 路径配置 ────────────────────
# 模型与嵌入根目录（根据实际环境修改）
MODELS=/root/autodl-fs/paper/RHO-main

# Ukraine 数据集根目录
DATA=/home/shu1004/lyx/RHO-main/ukrine/process_data

# 脚本所在目录（structure_ukraine）
SCRIPT_DIR=/home/shu1004/lyx/RHO-main/src/structure_ukraine

# 融合嵌入路径：fuse_embeddings.py 输出的 ent_rel_embeddings.pkl
# 若未融合，可直接使用 TransE 原始嵌入：
#   EMBEDPATH=${DATA}/train_transe/TransE_768_result/ent_rel_embeddings
EMBEDPATH=${DATA}/fused_embeddings/ent_rel_embeddings

# ──────────────────── 训练超参 ────────────────────
MODE="input"
STEP=100
DIM=768
TAU=5e-1

# ──────────────────── 输出目录 ────────────────────
OUTDIR=${SCRIPT_DIR}/ukraine_output_contrast_geometric_tau${TAU}
LOG_DIR=${SCRIPT_DIR}/logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_DIR}/train_ukraine_${TIMESTAMP}.log

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTDIR}"

echo "========================================"
echo "  Ukraine RHO 训练启动"
echo "  日志文件: ${LOG_FILE}"
echo "  输出目录: ${OUTDIR}"
echo "========================================"

nohup bash -c "
CUDA_VISIBLE_DEVICES=0 \
python ${SCRIPT_DIR}/run_summarization_attention_geometric_gate.py \
  --use_memory_bank=True \
  --memory_bank_mode='all' \
  --use_kg_embedding=True \
  --mode=${MODE} \
  --model_name_or_path=${MODELS}/bart_base \
  --cache_dir=${MODELS}/cache_dir \
  --text_column=history \
  --summary_column=response \
  --train_file=${DATA}/ukraine_train_dataset.csv \
  --validation_file=${DATA}/ukraine_valid_dataset.csv \
  --entity_relation_embedding_path=${EMBEDPATH} \
  --pad_to_max_length=True \
  --output_dir=${OUTDIR} \
  --learning_rate=3.5e-5 \
  --do_train \
  --logging_strategy=steps \
  --logging_steps=${STEP} \
  --logging_first_step \
  --logging_dir=${OUTDIR} \
  --max_source_length=800 \
  --max_target_length=64 \
  --do_eval=True \
  --evaluation_strategy=steps \
  --eval_steps=${STEP} \
  --eval_accumulation_steps=1 \
  --early_stop=True \
  --early_stopping_patience=5 \
  --per_device_eval_batch_size=4 \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=16 \
  --save_total_limit=5 \
  --report_to='none' \
  --load_best_model_at_end \
  --overwrite_cache \
  --overwrite_output_dir \
  --num_train_epochs=50 \
  --save_strategy=steps \
  --save_steps=${STEP} \
  --tau=${TAU}
" > "${LOG_FILE}" 2>&1 &

echo "后台进程 PID: $!"
echo "查看日志: tail -f ${LOG_FILE}"
