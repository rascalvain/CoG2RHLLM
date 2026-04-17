# 乌克兰数据集预处理流程说明

本文档介绍 `russian/process_data/` 目录下所有脚本的作用及服务器运行命令。  
脚本需按以下顺序依次执行。

---

## 整体流程

```
ukraine_opendialkg_pog.json
ukraine_opendialkg_pog_dialogs.json
        │
        ├─── [Step 1] process_data.py ──────────► ukraine_kg/（TransE 训练数据）
        │                                                  │
        │                                    [Step 2] run_train_transe.sh
        │                                                  │
        │                                    [Step 3] fuse_embeddings.py ──► fused_embeddings/
        │
        ├─── [Step 4] convert_ukraine.py ──────► ukraine_train/valid/test.txt
        │
        ├─── [Step 5] neg_sample_ukraine.py ───► ukraine_train_with_neg.txt
        │
        ├─── [Step 6] build_dataset_ukraine.py ► ukraine_train_dataset.csv
        │
        └─── [Step 7] build_eval_ukraine.py ───► ukraine_valid/test_dataset.csv
```

---

## Step 1 — `process_data.py`

**作用**：从原始 JSON 数据中提取知识图谱三元组，生成 TransE 训练所需的全套文件。

- 遍历每条对话中的 `action` 消息，读取 `metadata.path[1]` 中的三元组 `(head, relation, tail)`
- 对实体和关系名称进行 ASCII 规范化（去除 Unicode 特殊字符，防止 OpenKE C 库崩溃）
- 按 8:1:1 划分训练/验证/测试集（`--no_split` 模式可将全量作为训练集）
- 输出 `entity2id.txt`、`relation2id.txt`、`train2id.txt`、`valid2id.txt`、`test2id.txt`

**输出目录结构**：
```
ukraine_kg/
├── entity2id.txt          # 实体 → ID 映射（1008 个实体）
├── relation2id.txt        # 关系 → ID 映射（76 种关系）
├── train2id.txt           # 训练三元组（head_id\ttail_id\trel_id）
├── valid2id.txt           # 验证三元组
├── test2id.txt            # 测试三元组
└── ukraine_triples.txt    # 原始三元组文本（备用）
```

**运行命令**：

```bash
# 标准划分（8:1:1）
python process_data.py \
    --input_file  /home/shu1004/lyx/RHO-main/ukrine/data/ukraine_opendialkg_pog.json \
    --output_dir  /home/shu1004/lyx/RHO-main/ukrine/process_data/ukraine_kg

# 全量作为训练集（推荐，数据量小时使用）
python process_data.py \
    --input_file  /home/shu1004/lyx/RHO-main/ukrine/data/ukraine_opendialkg_pog.json \
    --output_dir  /home/shu1004/lyx/RHO-main/ukrine/process_data/ukraine_kg \
    --no_split
```

---

## Step 2 — `train_transe/run_train_transe.sh`

**作用**：使用 OpenKE 框架在乌克兰知识图谱上训练 TransE 嵌入，后台运行并按时间戳记录日志。

- 调用 `/home/shu1004/lyx/RHO-main/OpenKE/train_transe.py`
- 训练完成后输出 `ent_embeddings`、`rel_embeddings`（pickle 格式）

**注意事项**：
- `--datadir` 路径末尾**必须加 `/`**（OpenKE C 库直接拼接文件名）
- `--batch_size` 必须小于训练三元组总数（当前为 868，建议 64）

**运行命令**：

```bash
cd /home/shu1004/lyx/RHO-main/ukrine/process_data/train_transe
chmod +x run_train_transe.sh
./run_train_transe.sh

# 或直接运行（不使用 nohup）
CUDA_VISIBLE_DEVICES=0 python /home/shu1004/lyx/RHO-main/OpenKE/train_transe.py \
    --dim        768 \
    --lr         0.5 \
    --margin     5.0 \
    --outdir     /home/shu1004/lyx/RHO-main/ukrine/process_data/train_transe/TransE_768_result/ \
    --datadir    /home/shu1004/lyx/RHO-main/ukrine/process_data/ukraine_kg/ \
    --save_steps 10 \
    --batch_size 64 \
    --epoch      3000 \
    --patient    50

# 查看训练日志
tail -f /home/shu1004/lyx/RHO-main/ukrine/process_data/train_transe/logs/train_<时间戳>.log
```

---

## Step 3 — `fuse_embeddings.py`

**作用**：将 TransE 结构嵌入与 Sentence-Transformer 语义嵌入加权融合，生成高质量混合嵌入。

- TransE 嵌入捕捉图谱拓扑结构，但在小图谱上质量较低
- SBERT 嵌入捕捉实体名称语义，即使实体在图中只出现一次也有效
- 融合公式：`fused = α × norm(TransE) + (1-α) × norm(SBERT)`
- 额外保存各分量归一化结果，方便后续用不同 α 重新融合

**运行命令**：

```bash
# 安装依赖（如未安装）
pip install sentence-transformers

python fuse_embeddings.py \
    --kg_dir      /home/shu1004/lyx/RHO-main/ukrine/process_data/ukraine_kg/ \
    --transe_dir  /home/shu1004/lyx/RHO-main/ukrine/process_data/train_transe/TransE_768_result/ \
    --output_dir  /home/shu1004/lyx/RHO-main/ukrine/process_data/fused_embeddings/ \
    --model_name  all-mpnet-base-v2 \
    --alpha       0.3

# 调整权重（无需重跑 SBERT，直接用已保存的分量）
python -c "
import pickle, numpy as np
alpha = 0.4
transe = pickle.load(open('./fused_embeddings/ent_transe_norm', 'rb'))
sbert  = pickle.load(open('./fused_embeddings/ent_sbert_norm',  'rb'))
fused  = alpha * transe + (1 - alpha) * sbert
norms  = np.linalg.norm(fused, axis=1, keepdims=True)
fused  = fused / np.where(norms==0, 1, norms)
pickle.dump(fused, open('./fused_embeddings/ent_embeddings', 'wb'))
print('重新融合完成, alpha =', alpha)
"
```

---

## Step 4 — `convert_ukraine.py`

**作用**：将 `ukraine_opendialkg_pog_dialogs.json` 转换为 RHO 训练框架所需的 TXT 格式（与原 `convert_opendialkg.py` 输出格式完全一致）。

- 每条对话固定 8 个 turn：`用户问题 → [action+回答] × 3轮 → 最终总结`
- 每对 action+回答 生成一条训练样本，共 4 条/对话（3 条含 KB + 1 条空 KB）
- 按对话为单位划分 train/valid/test，防止同一对话的样本跨集出现
- 输出每行一个 JSON 字符串（`.txt` 格式）

**输出样本格式**：
```json
{
  "history":        [["user", "..."], ["assistant", "..."]],
  "response":       ["assistant", "..."],
  "knowledge_base": [["Donald Trump", "IS_IDENTIFIED_AS", "President"]],
  "dialogue_id":    "1"
}
```

**运行命令**：

```bash
BASE=/home/shu1004/lyx/RHO-main/ukrine/process_data
DATA=/home/shu1004/lyx/RHO-main/ukrine/data/ukraine_opendialkg_pog_dialogs.json

# 生成训练集（80%）
python convert_ukraine.py \
    --input_file  $DATA \
    --out_file    $BASE/ukraine_train.txt \
    --split       train

# 生成验证集（10%）
python convert_ukraine.py \
    --input_file  $DATA \
    --out_file    $BASE/ukraine_valid.txt \
    --split       valid

# 生成测试集（10%）
python convert_ukraine.py \
    --input_file  $DATA \
    --out_file    $BASE/ukraine_test.txt \
    --split       test

# 生成全量（不划分，用于查看/调试）
python convert_ukraine.py \
    --input_file  $DATA \
    --out_file    $BASE/ukraine_all.txt \
    --split       all
```

---

## Step 5 — `neg_sample_ukraine.py`

**作用**：对训练集进行负采样增量，为每条含 KB 的正样本生成对应的负样本（替换 `knowledge_base` 为错误 KB，`type=0`），同时为所有正样本添加 `type=1` 标记。

**负样本三级策略（`mixed` 模式按优先级依次尝试）**：

| 级别 | 策略 | 说明 |
|------|------|------|
| 硬（hard） | 同一对话其他轮次的 KB | 上下文相关但与当前 response 不匹配，最具挑战性 |
| 中（medium） | 共享实体但来自不同对话的 KB | 语义相近，有一定迷惑性 |
| 易（easy） | 全局随机 KB | 与原 `neg_sample_new.py` 行为一致 |

> **注意**：`knowledge_base` 为空的样本（最终总结轮，共 1138 条）仅保留正样本，不做负采样。

**运行命令**：

```bash
BASE=/home/shu1004/lyx/RHO-main/ukrine/process_data

# 默认：mixed 策略，每正样本 1 条负样本
python neg_sample_ukraine.py \
    --input_file  $BASE/ukraine_train.txt \
    --output_file $BASE/ukraine_train_with_neg.txt \
    --strategy    mixed \
    --neg_per_pos 1

# 生成更多负样本（每正样本 2 条，数据量翻倍）
python neg_sample_ukraine.py \
    --input_file  $BASE/ukraine_train.txt \
    --output_file $BASE/ukraine_train_with_neg_x2.txt \
    --strategy    mixed \
    --neg_per_pos 2
```

---

## Step 6 — `build_dataset_ukraine.py`

**作用**：将训练集 TXT 文件构建为 BART 模型可直接读取的 CSV 文件，核心工作是构建 `entity_relation_ids`（每个 token 对应的实体/关系 ID 标注）。

**处理流程**：
1. 读取三元组，通过 `entity2id`/`relation2id` 转为 ID，构建 `memory_bank`
2. 用 BART tokenizer 对知识前缀进行分词，生成 KB 段的 `entity_relation_ids`
3. 用 Sentence-Transformer + Flair NER 在对话历史中定位实体提及
4. 合并 KB 段和对话历史段的 ID 标注，校验长度一致性

**输出 CSV 列**：`entity_relation_ids | memory_bank | history | response | type`

**运行命令**：

```bash
BASE=/home/shu1004/lyx/RHO-main/ukrine/process_data
MODELS=/root/autodl-fs/paper/RHO-main

python build_dataset_ukraine.py \
    --file_entity_id   $BASE/ukraine_kg/entity2id.txt \
    --file_relation_id $BASE/ukraine_kg/relation2id.txt \
    --input_file       $BASE/ukraine_train_with_neg.txt \
    --out_file         $BASE/ukraine_train_dataset.csv \
    --bart_model       $MODELS/bart_base \
    --sbert_model      $MODELS/paraphrase-distilroberta-base-v2 \
    --max_hist_len     3 \
    --mod              all
```

---

## Step 7 — `build_eval_ukraine.py`

**作用**：将验证集和测试集 TXT 文件构建为 CSV，与 Step 6 逻辑完全一致，针对 eval 集的差异做以下适配：

| 差异点 | 训练集脚本 | 本脚本（eval）|
|--------|-----------|--------------|
| `type` 字段缺失 | `KeyError` 崩溃 | 默认为 `1` |
| 长度不一致 | `assert False` 崩溃 | 打印警告并跳过 |
| 非英文过滤 | 开启（训练集删去） | **关闭**（eval 不能删去） |
| 空 KB 样本 | 可选跳过 | 保留（全零标注） |

**运行命令**：

```bash
BASE=/home/shu1004/lyx/RHO-main/ukrine/process_data
MODELS=/root/autodl-fs/paper/RHO-main

# 验证集
python build_eval_ukraine.py \
    --file_entity_id   $BASE/ukraine_kg/entity2id.txt \
    --file_relation_id $BASE/ukraine_kg/relation2id.txt \
    --input_file       $BASE/ukraine_valid.txt \
    --out_file         $BASE/ukraine_valid_dataset.csv \
    --bart_model       $MODELS/bart_base \
    --sbert_model      $MODELS/paraphrase-distilroberta-base-v2 \
    --max_hist_len     3

# 测试集
python build_eval_ukraine.py \
    --file_entity_id   $BASE/ukraine_kg/entity2id.txt \
    --file_relation_id $BASE/ukraine_kg/relation2id.txt \
    --input_file       $BASE/ukraine_test.txt \
    --out_file         $BASE/ukraine_test_dataset.csv \
    --bart_model       $MODELS/bart_base \
    --sbert_model      $MODELS/paraphrase-distilroberta-base-v2 \
    --max_hist_len     3
```

---

## 数据统计

| 数据集 | 对话数 | 样本数（转换后） | 正样本 | 负样本 |
|--------|--------|----------------|--------|--------|
| 全量   | 1423   | 5692           | —      | —      |
| 训练集 | 1138   | 4552           | 4552   | 3414   |
| 验证集 | 142    | 568            | 568    | 0      |
| 测试集 | 143    | 572            | 572    | 0      |

> 训练集负采样后总样本数：**7966 条**（正 4552 + 负 3414）  
> 知识图谱：**1008 个实体**，**76 种关系**，**868 条唯一三元组**

---

## 文件目录结构（完成后）

```
russian/process_data/
├── process_data.py               # Step 1：KG 三元组提取
├── fuse_embeddings.py            # Step 3：嵌入融合
├── convert_ukraine.py            # Step 4：对话格式转换
├── neg_sample_ukraine.py         # Step 5：负采样增量
├── build_dataset_ukraine.py      # Step 6：训练集构建
├── build_eval_ukraine.py         # Step 7：验证/测试集构建
├── README.md                     # 本文档
│
├── ukraine_kg/                   # Step 1 输出
│   ├── entity2id.txt
│   ├── relation2id.txt
│   ├── train2id.txt
│   ├── valid2id.txt
│   ├── test2id.txt
│   └── ukraine_triples.txt
│
├── train_transe/                 # Step 2 脚本 & 输出
│   ├── run_train_transe.sh
│   ├── TransE_768_result/
│   │   ├── ent_embeddings
│   │   └── rel_embeddings
│   └── logs/
│
├── fused_embeddings/             # Step 3 输出
│   ├── ent_embeddings
│   ├── rel_embeddings
│   ├── ent_transe_norm
│   ├── ent_sbert_norm
│   ├── rel_transe_norm
│   └── rel_sbert_norm
│
├── ukraine_train.txt             # Step 4 输出
├── ukraine_valid.txt
├── ukraine_test.txt
├── ukraine_train_with_neg.txt    # Step 5 输出
├── ukraine_train_dataset.csv     # Step 6 输出
├── ukraine_valid_dataset.csv     # Step 7 输出
└── ukraine_test_dataset.csv      # Step 7 输出
```
