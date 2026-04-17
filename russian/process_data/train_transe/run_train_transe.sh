#!/bin/bash

# TransE 训练脚本 —— 乌克兰数据集（全量训练，--no_split 模式）
# 日志输出至 logs/ 目录，按时间戳命名

BASE_DIR="/home/shu1004/lyx/RHO-main/ukrine/process_data/train_transe"
LOG_DIR="${BASE_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "启动 TransE 训练，日志写入: ${LOG_FILE}"

nohup bash -c "
CUDA_VISIBLE_DEVICES=0 python /home/shu1004/lyx/RHO-main/OpenKE/train_transe.py \
  --dim        768 \
  --lr         0.5 \
  --margin     5.0 \
  --outdir     ${BASE_DIR}/TransE_768_result/ \
  --datadir    /home/shu1004/lyx/RHO-main/ukrine/process_data/ukraine_kg/ \
  --save_steps 10 \
  --batch_size 64 \
  --epoch      3000 \
  --patient    50
" > "${LOG_FILE}" 2>&1 &

echo "后台进程 PID: $!"
echo "查看日志: tail -f ${LOG_FILE}"
