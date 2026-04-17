#!/bin/bash
MODE="input"
DATAMODE="all"
STEP=100
DIM=768
EMBEDPATH=/autodl-fs/data/RHO-main/OpenKE/TransE_768_result/ent_rel_embeddings

# IFMEMORY="no_memory"
IFMEMORY='entity_memory'
# need to change --use_memory_bank

IFKG="_no_KG"
# IFKG=''
# need to change --use_kg_embedding

OUTDIR=$MODE"_"$IFMEMORY$IFKG"_output1"


CUDA_VISIBLE_DEVICES=0 \
python run_summarization.py \
  --use_memory_bank=True \
  --memory_bank_mode="entity" \
  --use_kg_embedding=False \
  --model_name_or_path="/autodl-fs/data/RHO-main/bart_base" \
  --text_column=history \
  --summary_column=response \
  --train_file=data/train.csv \
  --validation_file=data/val.csv \
  --entity_relation_embedding_path=$EMBEDPATH \
  --pad_to_max_length=True \
  --mode=$MODE \
  --output_dir=$OUTDIR \
  --learning_rate=3.5e-5 \
  --do_train \
  --do_eval \
  --evaluation_strategy=steps \
  --eval_steps=$STEP \
  --logging_strategy=steps \
  --logging_steps=$STEP \
  --logging_first_step \
  --logging_dir=$OUTDIR \
  --max_source_length=800 \
  --max_target_length=64 \
  --per_device_train_batch_size=4 \
  --per_device_eval_batch_size=4 \
  --gradient_accumulation_steps=8 \
  --eval_accumulation_steps=1 \
  --early_stop=True \
  --early_stopping_patience=3 \
  --save_total_limit=1 \
  --load_best_model_at_end \
  --overwrite_cache \
  --overwrite_output_dir \
  --num_train_epochs=50 \
  >train_rho.log \
  2>&1 \
  &
