#!/bin/bash

# 验证集 + 测试集 CSV 构建（与 run_build_dataset.sh 路径风格一致）
BASE=/home/shu1004/lyx/RHO-main/ukrine/process_data
MODELS=/root/autodl-fs/paper/RHO-main
SCRIPT_DIR=/home/shu1004/lyx/RHO-main/ukrine/process_data
LOG_DIR=${SCRIPT_DIR}/logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_DIR}/build_eval_${TIMESTAMP}.log

mkdir -p "${LOG_DIR}"

echo "启动验证集 / 测试集构建，日志写入: ${LOG_FILE}"

nohup bash -c "
echo '========== 验证集 ukraine_valid_dataset.csv =========='
python ${SCRIPT_DIR}/build_eval_ukraine.py \
    --file_entity_id   ${BASE}/ukraine_kg/entity2id.txt \
    --file_relation_id ${BASE}/ukraine_kg/relation2id.txt \
    --input_file       ${BASE}/ukraine_valid.txt \
    --out_file         ${BASE}/ukraine_valid_dataset.csv \
    --bart_model       ${MODELS}/bart_base \
    --sbert_model      ${MODELS}/paraphrase-distilroberta-base-v2 \
    --ner_model        ner \
    --max_hist_len     3 \
    --mod              all

echo ''
echo '========== 测试集 ukraine_test_dataset.csv =========='
python ${SCRIPT_DIR}/build_eval_ukraine.py \
    --file_entity_id   ${BASE}/ukraine_kg/entity2id.txt \
    --file_relation_id ${BASE}/ukraine_kg/relation2id.txt \
    --input_file       ${BASE}/ukraine_test.txt \
    --out_file         ${BASE}/ukraine_test_dataset.csv \
    --bart_model       ${MODELS}/bart_base \
    --sbert_model      ${MODELS}/paraphrase-distilroberta-base-v2 \
    --ner_model        ner \
    --max_hist_len     3 \
    --mod              all

echo ''
echo '========== 全部完成 =========='
" > "${LOG_FILE}" 2>&1 &

echo "后台进程 PID: $!"
echo "查看日志: tail -f ${LOG_FILE}"
