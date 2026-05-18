# CoG2 RH-LLM (Chapter 2)

This repository is the implementation for the Chapter 2 project, built on top of [RHO: Reducing Hallucination in Open-domain Dialogues with Knowledge Grounding](https://aclanthology.org/2023.findings-acl.275/).

The main improved framework in this codebase is:

- Geometric weighted graph attention for fine-grained triple fusion
- Knowledge gating for suppressing noisy KG signals
- Graph-context contrastive learning for factual alignment

Reference manuscript in local workspace:

- `G:\小论文\第二章\ESWA\paper.pdf`

## 1. What Is New in This Repo

Compared with the original RHO pipeline, this project adds and integrates:

1. Geometric graph attention
- Core implementation: `src/model_structure/contrast_geometric_gate_update/modeling_bart_contrast_attention_geometric_gate.py`
- Main modules: `AdaptiveEpsilonGeometricMean`, `GraphAttention`, `GatingMechanism`

2. Contrastive objective
- Core implementation: `src/model_structure/contrast_geometric_gate_update/trainer_seq2seq.py`
- Contrastive temperature hyper-parameter: `--tau`

3. Extended training/prediction entrypoint
- `src/model_structure/contrast_geometric_gate_update/run_summarization_attention_geometric_gate.py`

## 2. Reported Results (from paper)

On OpenDialKG, the manuscript reports the following (Table 1):

| Model | BLEU-4 | ROUGE-L | QuestEval-RD | QuestEval-RF |
|---|---:|---:|---:|---:|
| RHO | 20.77 | 39.54 | 48.41 | 43.84 |
| CoG2 RH-LLM (full) | 20.81 | 41.97 | 49.96 | 45.42 |

Note:

- ROUGE-L gain vs RHO: +2.43 absolute (paper also reports relative gains in abstract/conclusion sections)
- Entity coverage precision in paper is reported around 97.5%

## 3. Repository Map

Main directories:

- `src/model_structure/contrast_geometric_gate_update/`: full improved model (recommended)
- `src/model_structure/normal/`: ablation-like variant
- `src/model_structure/structure_ukraine/`: scripts/configs for Ukraine-domain experiments
- `api/`: Flask inference API server (wraps the trained model for online prediction)
- `src/data/`: OpenDialKG data processing scripts and processed CSVs
- `src/OpenKE/`: KG embedding (TransE/TransH/RotatE/ComplEx wrappers)
- `negative_sampling_new/`: extra negative-sampling utilities
- `kg-cruse_/`: reranking module related code
- `KG-BART/`: baseline code
- `metrics/`: entity coverage script and evaluation helpers

## 4. Environment Setup

Recommended:

```bash
conda create -n cog2_rh python=3.7 -y
conda activate cog2_rh
pip install -r requirements.txt
```

Optional but commonly needed:

```bash
python -m spacy download en_core_web_sm
```

Notes:

- First run will download model weights for BART / Sentence-Transformer / Flair NER.
- In `run_summarization_attention_geometric_gate.py`, `HF_ENDPOINT` is set to `https://hf-mirror.com`.

## 5. OpenDialKG Reproduction

### 5.1 Prepare raw OpenDialKG

If you do not already have the raw data:

```bash
git clone https://github.com/facebookresearch/opendialkg.git
```

Or directly use local files under `src/opendialkg/data/` if already prepared.

### 5.2 Build KG embeddings with OpenKE

Compile OpenKE backend:

```bash
cd src/OpenKE/openke
bash make.sh
cd ..
```

Prepare OpenDialKG triples for OpenKE:

```bash
python process_data.py \
  --input_dir ../opendialkg/data \
  --output_dir opendialkg

python n-n.py \
  --input_dir opendialkg \
  --output_dir opendialkg \
  --test_path opendialkg/test2id.txt
```

Train TransE:

```bash
CUDA_VISIBLE_DEVICES=0 python train_transe.py \
  --dim 768 \
  --lr 0.5 \
  --margin 5.0 \
  --outdir TransE_768_result \
  --datadir ./opendialkg \
  --save_steps 10 \
  --batch_size 4096 \
  --epoch 1000 \
  --patient 10
```

Merge entity + relation embeddings:

```bash
python merge.py --input_dir TransE_768_result
```

Expected merged embedding path:

- `src/OpenKE/TransE_768_result/ent_rel_embeddings`

### 5.3 Build dialogue training CSVs

```bash
cd ../data

python convert_opendialkg.py \
  --input_file ../opendialkg/data/opendialkg.csv \
  --out_file processed_opendialkg.txt

python filter.py \
  --input_file processed_opendialkg.txt \
  --out_file only_path.txt

python build_dataset.py \
  --input_file only_path.txt \
  --out_file all_3.csv \
  --file_entity_id ../OpenKE/opendialkg/entity2id.txt \
  --file_relation_id ../OpenKE/opendialkg/relation2id.txt

python split.py --input_file all_3.csv
```

Generated files:

- `src/data/train.csv`
- `src/data/val.csv`
- `src/data/test.csv`

Data format (CSV columns):

- `entity_relation_ids`
- `memory_bank`
- `history`
- `response`

### 5.4 Train the improved CoG2 RH-LLM

```bash
cd ../model_structure/contrast_geometric_gate_update

EMBEDPATH=../../OpenKE/TransE_768_result/ent_rel_embeddings
OUTDIR=./outputs/cog2_tau05

CUDA_VISIBLE_DEVICES=0 python run_summarization_attention_geometric_gate.py \
  --use_memory_bank=True \
  --memory_bank_mode=all \
  --use_kg_embedding=True \
  --mode=input \
  --model_name_or_path=facebook/bart-base \
  --text_column=history \
  --summary_column=response \
  --train_file=../../data/train.csv \
  --validation_file=../../data/val.csv \
  --entity_relation_embedding_path=$EMBEDPATH \
  --pad_to_max_length=True \
  --output_dir=$OUTDIR \
  --learning_rate=3.5e-5 \
  --do_train \
  --do_eval \
  --evaluation_strategy=steps \
  --eval_steps=100 \
  --logging_strategy=steps \
  --logging_steps=100 \
  --max_source_length=800 \
  --max_target_length=64 \
  --per_device_train_batch_size=4 \
  --per_device_eval_batch_size=4 \
  --gradient_accumulation_steps=16 \
  --save_total_limit=5 \
  --load_best_model_at_end \
  --overwrite_cache \
  --overwrite_output_dir \
  --num_train_epochs=50 \
  --save_strategy=steps \
  --save_steps=100 \
  --tau=5e-1
```

### 5.5 Predict on test split

```bash
EMBEDPATH=../../OpenKE/TransE_768_result/ent_rel_embeddings
CKPT=./outputs/cog2_tau05

CUDA_VISIBLE_DEVICES=0 python run_summarization_attention_geometric_gate.py \
  --use_memory_bank=True \
  --memory_bank_mode=all \
  --use_kg_embedding=True \
  --mode=input \
  --model_name_or_path=$CKPT \
  --text_column=history \
  --summary_column=response \
  --test_file=../../data/test.csv \
  --train_file=../../data/train.csv \
  --entity_relation_embedding_path=$EMBEDPATH \
  --pad_to_max_length=True \
  --output_dir=$CKPT \
  --do_predict=True \
  --predict_with_generate=True \
  --num_beams=4 \
  --max_source_length=800 \
  --max_target_length=64 \
  --per_device_eval_batch_size=4 \
  --generation_file=B4_generated_predictions.txt \
  --overwrite_cache
```

Output:

- `$CKPT/B4_generated_predictions.txt`

## 6. Ukraine-Domain Pipeline (Optional)

The repository also includes a complete Ukraine-domain workflow:

- Data processing scripts: `russian/process_data/`
- Training/prediction scripts: `src/model_structure/structure_ukraine/`

Suggested order:

1. `process_data.py`
2. `train_transe/run_train_transe.sh`
3. `fuse_embeddings.py`
4. `convert_ukraine.py`
5. `neg_sample_ukraine.py`
6. `build_dataset_ukraine.py`
7. `build_eval_ukraine.py`
8. `run_train_ukraine.sh` and `run_predict_ukraine.sh`

Important:

- Some `.sh` files contain absolute paths; update them before running.

## 7. Evaluation

### 7.1 BLEU / ROUGE

- Already computed in training/prediction scripts via HuggingFace `datasets`.

### 7.2 Entity coverage

```bash
python metrics/spacy_ner.py \
  --source_file <source_file> \
  --target_file <target_file> \
  --generated_file <generated_file>
```

### 7.3 QuestEval / FeQA

- QuestEval and FeQA are not hardwired into the main training script.
- Use external tools under `metrics/` according to your evaluation protocol.

## 8. Inference API Server

A Flask-based REST API that wraps the trained Ukraine-domain model for online inference. The response structure is aligned with the prototype system's `page3_pipeline.Page3PipelineService.run_prediction()`.

### 8.1 Directory

```
api/
├── config.py           # Paths and hyper-parameters (edit before first run)
├── data_processor.py   # Builds entity_relation_ids / memory_bank from text + triples
├── inference.py        # Model loading and generate()
├── response_builder.py # Packs BART output into page3_pipeline-compatible structure
└── app.py              # Flask routes: POST /predict, GET /health
```

### 8.2 Configuration

Open `api/config.py` and fill in:

| Variable | Description |
|---|---|
| `MODEL_CHECKPOINT_DIR` | Directory of the trained checkpoint (contains `pytorch_model.bin`, `config.json`, etc.) |
| `KG_EMBEDDING_PATH` | Path to the merged `ent_rel_embeddings` pickle file (no extension) |

Other paths (`ENTITY2ID_PATH`, `RELATION2ID_PATH`) point to `russian/process_data/ukraine_kg/` by default.

### 8.3 Start the server

```bash
cd api
python app.py
```

The model is loaded once at startup. The server listens on `0.0.0.0:5000` by default.

### 8.4 API endpoints

**POST /predict**

Request body (JSON):

```json
{
    "query":   "Target behavior reasoning query",
    "history": "Given the knowledge: ukraine<sep> located in<sep> europe<triple> donald trump<sep> intends to<sep> ukraine<user> What is the latent behavior?<assistant>",
    "triples": [["ukraine", "located_in", "europe"], ["donald trump", "intends_to", "ukraine"]]
}
```

| Field | Type | Description |
|---|---|---|
| `query` | string | Query text shown in DIALOG and used for report generation |
| `history` | string | Formatted dialogue context with `<sep>`, `<triple>`, `<user>`, `<assistant>` tokens |
| `triples` | list | Subgraph triples `[[head, relation, tail], ...]` |

Response (JSON): same structure as `page3_pipeline.run_prediction()`:

```
DIALOG / METRICS / REASONING_FRAMEWORK / FINAL_ASSESSMENT / STRATEGIES
PROMPT_PREFIX / TRANSE_EMBEDDING / meta
```

The BART model output is placed in `FINAL_ASSESSMENT.text` and `DIALOG[7].text` (assess role). All other fields are generated by deterministic rules identical to the fallback path in `page3_pipeline`.

**GET /health**

Returns `{"status": "ok", "model_loaded": true}` when the model is ready.

### 8.5 Notes

- `history` must follow the exact token format used during training (see Section 6 for data building).
- Triple entity/relation names are matched against `entity2id.txt` / `relation2id.txt` after ASCII normalization. Unknown names are silently skipped.
- Dialogue history entity IDs are simplified to zeros (no NER matching at inference time).

## 9. Practical Notes

1. `memory_bank` size constraint
- In current preprocessing logic, each sample is expected to have at most 2 triples in `memory_bank`.
- Samples with more than 2 triples can trigger assertion errors.

2. Contrastive learning behavior
- Training uses `preprocess_function_neg` (negative pairs are constructed during preprocessing).
- Temperature is controlled by `--tau`.

3. Baseline entrypoints are still available
- Original RHO-like pipeline: `src/run_summarization.py`, `src/generation.py`
- Rerank-related code: `kg-cruse_/`
- KG-BART baseline: `KG-BART/`

## 10. Citation

If this repository helps your work, please cite the original RHO paper and the Chapter 2 manuscript.

RHO:

```bibtex
@inproceedings{ji2023rho,
  title={RHO: Reducing hallucination in open-domain dialogues with knowledge grounding},
  author={Ji, Ziwei and Liu, Zihan and Lee, Nayeon and Yu, Tiezheng and Wilie, Bryan and Zeng, Min and Fung, Pascale},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  pages={4504--4522},
  year={2023}
}
```

For the improved CoG2 RH-LLM citation, please use the bibliographic metadata in:

- `G:\小论文\第二章\ESWA\paper.pdf`
