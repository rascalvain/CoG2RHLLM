# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler,DataLoader, RandomSampler, DistributedSampler,SequentialSampler
from transformers.trainer_pt_utils import IterableDatasetShard, LengthGroupedSampler, DistributedLengthGroupedSampler, \
    DistributedSamplerWithLoop
from transformers.file_utils import is_datasets_available
from transformers.training_args import ParallelMode
import numpy as np
if is_datasets_available():
    import datasets

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
# from trainer import Trainer
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging
import torch.nn.functional as F

import pickle

_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast
    _is_torch_generator_available = True

logger = logging.get_logger(__name__)

class BalancedSampler(Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset, pos_ratio: float = 1/3,batch_size: int = 32):
        """
        初始化采样器
        :param data_source: Dataset 对象
        :param pos_ratio: 每个 batch 中正样本的比例，默认为 0.5
        """
        self.data_source = data_source
        self.pos_ratio = pos_ratio
        self.batch_size = batch_size
        print('批量大小为',self.batch_size)
        # 获取所有标签，假设标签为 0 或 1
        self.labels = np.array([sample['neg_pos_label'] for sample in data_source])  # 提取标签数组

        # 获取正样本和负样本的索引
        self.pos_indices = np.where(self.labels == 1)[0]  # 正样本的索引
        self.neg_indices = np.where(self.labels == 0)[0]  # 负样本的索引
        print("初始化该采样器")
        print(f"Positive samples: {len(self.pos_indices)}, Negative samples: {len(self.neg_indices)}")
    def __iter__(self):
        """
        生成平衡的批次索引
        """
        # 计算每个 batch 中正样本和负样本的数量
        # batch_size = len(self.data_source)  # 假设 batch_size 等于数据集大小
        num_batches = len(self.data_source) // self.batch_size

        all_indices = []
        for _ in range(num_batches):
            num_pos_samples = int(self.batch_size * self.pos_ratio)  # 需要的正样本数
            num_neg_samples = self.batch_size - num_pos_samples  # 需要的负样本数
            # 确保负样本的数量不超过负样本总数
            num_neg_samples = min(num_neg_samples, len(self.neg_indices))
            num_pos_samples = min(num_pos_samples, len(self.pos_indices))
            # print(f"num_pos_samples:{num_pos_samples},num_neg_samples:{num_neg_samples}")
            # 如果正负样本不足以填充批次，则给出警告
            # 注意：与 int() 截断后的目标值比较，避免 batch_size 不能被 1/pos_ratio 整除时产生误报
            target_pos = int(self.batch_size * self.pos_ratio)
            target_neg = self.batch_size - target_pos
            if num_neg_samples < target_neg:
                print(f"Warning: Not enough negative samples to meet the requested ratio in this batch.")
            if num_pos_samples < target_pos:
                print(f"Warning: Not enough positive samples to meet the requested ratio in this batch.")

            # 随机选取正样本和负样本
            pos_indices = np.random.choice(self.pos_indices, num_pos_samples, replace=False)
            neg_indices = np.random.choice(self.neg_indices, num_neg_samples, replace=False)

            # 合并正负样本的索引
            indices = np.concatenate([pos_indices, neg_indices])
            # np.random.shuffle(indices)  # 打乱顺序
            # indices = [int(i) for i in indices]
            # print(f"Batch sample labels: {[self.labels[i] for i in indices[:6]]}")
            all_indices.append(indices)
            # print(f"Batch sample: {pos_indices} positive, {neg_indices} negative")
        # random.shuffle(all_indices)
        # print(f"Epoch sampling indices: {all_indices}")
        # print(f"Epoch sampling indices: {[self.data_source[i] for i in all_indices[:6]]}")
        # print(f"Epoch sampling indices: {[self.labels[i] for i in all_indices[:6]]}")
        np.random.shuffle(all_indices)
        all_indices_flat = np.concatenate(all_indices)
        all_indices = [int(i) for i in all_indices_flat]
        # 返回采样器迭代器
        # print(f"Epoch sampling indices: {[self.labels[i] for i in all_indices[:60]]}")
        return iter(all_indices)

    def __len__(self):
        """
        返回数据集的长度（即采样的样本数）
        """
        return len(self.data_source) // self.batch_size

class Seq2SeqTrainer(Trainer):
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        self._max_length = max_length
        self._num_beams = num_beams
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
            self,
            test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
            # do_sample: Optional[bool] = None,
            # top_k: Optional[int] = None,
            # top_p: Optional[float] = None,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        .. note::

            If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        self._max_length = max_length
        self._num_beams = num_beams
        # self._do_sample = do_sample
        # self._top_k = top_k
        # self._top_p = top_p
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        # has_kb 不属于 model.forward() 参数，在调用 generate/forward 之前弹出
        inputs.pop('has_kb', None)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            # "do_sample": self._do_sample,
            # "top_k": self._top_k,
            # "top_p": self._top_p,
        }
        generated_tokens = self.model.generate(
            inputs["input_ids"],
            input_entity_relation_ids=inputs["input_entity_relation_ids"],
            memory_bank=inputs["memory_bank"],
            memory_bank_attention_mask=inputs["memory_bank_attention_mask"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is None:
            raise ValueError(
                f"Tensor need to be padded to `max_length={max_length}` but no tokenizer was passed when creating "
                "this `Trainer`. Make sure to create your `Trainer` with the appropriate tokenizer."
            )
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None
        # print('执行此函数获取采样器')
        generator = None
        # if self.args.world_size <= 1 and _is_torch_generator_available:
        #     generator = torch.Generator()
        #     generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=self.args.seed,
                )

        else:
            if self.args.world_size <= 1:#如果只有一个训练设备
                # if _is_torch_generator_available:
                #     #修改，返回一个顺序采样器
                #     # return RandomSampler(self.train_dataset, generator=generator)
                # return SequentialSampler(self.train_dataset)
            #改为自己定义的采样器
                return BalancedSampler(self.train_dataset,batch_size=self.args.train_batch_size)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.dataset.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                shuffle=False
            )

        train_sampler = self._get_train_sampler()
        # print('采样器类别是：',type(train_sampler))
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False
        )

    def generate_mapping(self,pos_indices, num_neg_samples_per_pos=2):
        """
        生成正样本到负样本的映射字典。

        :param pos_indices: 正样本索引列表
        :param num_neg_samples_per_pos: 每个正样本对应的负样本数量，默认2
        :return: 正样本到负样本的映射字典
        """
        neg_indices = list(range(len(pos_indices) * num_neg_samples_per_pos))  # 生成负样本索引
        mapping = {}

        neg_idx = 0
        for pos_idx in pos_indices:
            # 每个正样本对应 num_neg_samples_per_pos 个负样本
            mapping[pos_idx] = neg_indices[neg_idx:neg_idx + num_neg_samples_per_pos]
            neg_idx += num_neg_samples_per_pos

        return mapping
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # has_kb 是对比学习的屏蔽标记，不属于 model.forward() 的参数，需提前弹出
        has_kb = inputs.pop('has_kb', None)
        outputs = model(**inputs)
        # print('输出打印：',outputs)
        is_train_flag = False
        ################################################################################ 1.进行对比学习损失的计算，当前思路是，获取memory_bank的子图向量，与decoder的hidden states进行比较
        if self.is_in_train:#如果是训练阶段
            # print('输入为：', inputs)
            neg_pos_label = inputs['neg_pos_label']
            neg_idx = []
            pos_idx = []
            # 得到一个batch中每个位置对应的是正样本还是负样本
            # 同时过滤掉 has_kb=0 的空 KB 样本（其 subgraph_embedding 为全零，参与对比损失无意义）
            for i in range(len(neg_pos_label)):
                if has_kb is not None and has_kb[i] == 0:
                    continue  # 空 KB 样本跳过对比学习
                if neg_pos_label[i] == 0:
                    neg_idx.append(i)
                else:
                    pos_idx.append(i)
            if len(neg_idx) != 0:
                is_train_flag = True
                mapping = self.generate_mapping(pos_indices=pos_idx, num_neg_samples_per_pos=2)
                decoder_last_hidden_state = outputs['encoder_last_hidden_state'] if isinstance(outputs, dict) else outputs[
                    2]  # 获取隐藏状态 [4,800,768] 分别对应batch_size,输出序列长度，每个token对应的维度
                # 分别获取正样本和负样本生成的hidden_state
                neg_hidden_state = decoder_last_hidden_state[neg_idx]
                pos_hidden_state = decoder_last_hidden_state[pos_idx]

                # 需要计算隐藏状态的平均表示
                pos_hidden_state_mean = torch.mean(pos_hidden_state, dim=1)
                neg_hidden_state_mean = torch.mean(neg_hidden_state, dim=1)

                # 获取子图嵌入的内容
                subgraph_embedding = outputs['subgraph_embedding'] if isinstance(outputs, dict) else outputs[-1]
                neg_subgraph_embedding = subgraph_embedding[neg_idx]
                pos_subgraph_embedding = subgraph_embedding[pos_idx]
                #########################################################################对比学习损失的计算逻辑
                tau = self.args.tau  # 获取温度参数
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                contrast_loss=torch.zeros((1),device=device)
                # 计算余弦相似度
                sclar=0.5
                for pos, neg_list in mapping.items():
                    pos_hidden_tmp = pos_hidden_state_mean[pos]
                    pos_subgraph_tmp = pos_subgraph_embedding[pos]
                    cos_sim1 = F.cosine_similarity(pos_hidden_tmp.unsqueeze(0), pos_subgraph_tmp.unsqueeze(0), dim=1) / tau
                    fenzi_tmp =torch.exp(cos_sim1)
                    fenmu1_sum = torch.zeros_like(cos_sim1)
                    fenmu2_sum = torch.zeros_like(cos_sim1)
                    for j in neg_list:
                        neg_hidden_tmp = neg_hidden_state_mean[j]
                        neg_subgraph_tmp = neg_subgraph_embedding[j]
                        cos_sim2 = F.cosine_similarity(neg_hidden_tmp.unsqueeze(0), pos_subgraph_tmp.unsqueeze(0), dim=1) / tau
                        cos_sim3 = F.cosine_similarity(pos_hidden_tmp.unsqueeze(0), neg_subgraph_tmp.unsqueeze(0), dim=1) / tau
                        fenmu1_sum+= torch.exp(cos_sim2)
                        fenmu2_sum+= torch.exp(cos_sim3)
                    contrast_loss_tmp = sclar*(torch.log(fenzi_tmp/fenmu1_sum)+sclar*torch.log(fenzi_tmp/fenmu2_sum))
                    contrast_loss += contrast_loss_tmp
                # for i in range(len(pos_idx)):
                #     pos_hidden_tmp = pos_hidden_state_mean[i]
                #     pos_subgraph_tmp = pos_subgraph_embedding[i]
                #     cos_sim1 = F.cosine_similarity(pos_hidden_tmp.unsqueeze(0), pos_subgraph_tmp.unsqueeze(0), dim=1) / tau
                #     cos_sim1 =torch.exp(cos_sim1)
                #     cos_sim2_sum = torch.zeros_like(cos_sim1)
                #     cos_sim3_sum = torch.zeros_like(cos_sim1)
                #     for j in range(len(neg_idx)):
                #         neg_hidden_tmp = neg_hidden_state_mean[j]
                #         neg_subgraph_tmp = neg_subgraph_embedding[j]
                #         cos_sim2 = F.cosine_similarity(neg_hidden_tmp.unsqueeze(0), pos_subgraph_tmp.unsqueeze(0), dim=1) / tau
                #         cos_sim3 = F.cosine_similarity(pos_hidden_tmp.unsqueeze(0), neg_subgraph_tmp.unsqueeze(0), dim=1) / tau
                #         cos_sim2_sum+= torch.exp(cos_sim2)
                #         cos_sim3_sum+= torch.exp(cos_sim3)

                # for i in range(len(neg_idx)):
                #     neg_hidden_tmp = neg_hidden_state_mean[i]
                #     pos_hidden_tmp = pos_hidden_state_mean[i]
                #     neg_subgraph_tmp = neg_subgraph_embedding[i]
                #     pos_subgraph_tmp = pos_subgraph_embedding[i]
                #     cos_sim1 = F.cosine_similarity(pos_hidden_tmp.unsqueeze(0), pos_subgraph_tmp.unsqueeze(0), dim=1) / tau#[1]
                #     cos_sim2 = F.cosine_similarity(neg_hidden_tmp.unsqueeze(0), pos_subgraph_tmp.unsqueeze(0), dim=1) / tau#[1]
                #     cos_sim3 = F.cosine_similarity(pos_hidden_tmp.unsqueeze(0), neg_subgraph_tmp.unsqueeze(0), dim=1) / tau#[1]
                #     contrast_loss_tmp = sclar*(torch.exp(cos_sim1)/torch.exp(cos_sim2))+sclar*(torch.exp(cos_sim1)/torch.exp(cos_sim3))
                #     contrast_loss+=contrast_loss_tmp
        #########################################################################
        ################################################################################
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if is_train_flag:#如果是训练阶段
            # print("如果是训练阶段，则加上对比学习损失！")
            contrast_loss=contrast_loss.squeeze()
            # print('最终计算出来的对比损失为：',contrast_loss,'grad_fn：',contrast_loss.grad_fn)
            # print('模型原有的损失为：',loss,'grad_fn：',contrast_loss.grad_fn)
        ######################################################### 两个loss进行求和，得到最终的loss
            final_loss=contrast_loss+loss
            # print('两个模型加和所得的最终损失为：',final_loss,'grad_fn：',final_loss.grad_fn)
            return (final_loss, outputs) if return_outputs else final_loss
        return (loss, outputs) if return_outputs else loss