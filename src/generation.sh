#!/bin/bash
MODE="input"
EMBEDPATH=/autodl-fs/data/RHO-main/OpenKE/TransE_768_result/ent_rel_embeddings
OUTDIR=/autodl-fs/data/RHO-main/src/input_entity_memory_no_KG_output1/checkpoint-100
GENERTYPE="B4"

CUDA_VISIBLE_DEVICES=0 python generation.py \
--use_memory_bank=True \
--memory_bank_mode="entity" \
--use_kg_embedding=True \
--model_name_or_path=$OUTDIR \
--text_column=history \
--summary_column=response \
--test_file="data/test.csv" \
--output_dir=$OUTDIR \
--entity_relation_embedding_path=$EMBEDPATH \
--pad_to_max_length=True \
--mode=$MODE \
--num_beams=4 \
--num_return_sequences=4 \
--predict_with_generate=True \
--max_source_length=800 \
--max_target_length=64 \
--per_device_eval_batch_size=8 \
--generation_path="/autodl-fs/data/RHO-main/src/rerank/generation_"$GENERTYPE".json" \
  >generation_rho.log \
  2>&1 \
  &