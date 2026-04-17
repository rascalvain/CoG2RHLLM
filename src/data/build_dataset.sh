#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 python build_dataset.py \
--input_file=only_path.txt \
--out_file=all_3.csv \
--file_entity_id=/autodl-fs/data/RHO-main/OpenKE/opendialkg/entity2id.txt \
--file_relation_id=/autodl-fs/data/RHO-main/OpenKE/opendialkg/relation2id.txt