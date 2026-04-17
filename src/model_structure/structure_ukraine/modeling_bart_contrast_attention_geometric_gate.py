# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch BART model. """
import copy
import math
import random
import warnings
from typing import Optional, Tuple
import pickle
from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers import PreTrainedModel
from transformers.utils import logging
from transformers import BartConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # See all BART models at https://huggingface.co/models?filter=bart
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


####################################################################################################
def entity_expand_attention_mask(mask: torch.Tensor, input_entity_relation_ids: torch.Tensor,
                                 original_attention_mask: torch.Tensor, dtype: torch.dtype):  # 扩展到和原始的注意力掩码的长度一致
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()  # batch size是第0维大小，src len是第一维大小
    tgt_len = original_attention_mask.size()[-1]  # 目标序列的长度，原始注意力掩码的最后一维长度
    assert input_entity_relation_ids.size()[-1] == original_attention_mask.size()[-1]

    expanded_entity_mask = input_entity_relation_ids[:, None, :, None].expand(bsz, 1, tgt_len, src_len).to(dtype)
    # [:, None, :, None]相当于是将这个张量reshape了一下，变成[bsz,1,src_seq_len,1](在外面多套了一层括号)
    # .expand(bsz, 1, tgt_len, src_len) 主要是在后两个维度做操作，最里层把每位分词对应的id都重复一遍，扩展为知识库的大小，便于后续计算全局知识基础时是否在这个位置进行替换
    # print("expanded_entity_mask", expanded_entity_mask, ~expanded_entity_mask.bool())

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    # print("inverted_mask", inverted_mask, inverted_mask.bool())

    expanded_entity_mask = torch.logical_or(inverted_mask.bool(), ~expanded_entity_mask.bool())
    # print("expanded_entity_mask", expanded_entity_mask)

    return inverted_mask.masked_fill(expanded_entity_mask, torch.finfo(dtype).min)


####################################################################################################
"""
设计一个可学习的epsilon出来，通过学习出来的epsilon去进行几何平均的计算
"""
class AdaptiveEpsilonGeometricMean(nn.Module):
    def __init__(self, init_epsilon=1e-5, reset_threshold=1e-7):
        super().__init__()
        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(init_epsilon)))
        self.init_epsilon = init_epsilon
        self.reset_threshold = reset_threshold

    def forward(self, attn_weights_head, attn_weights_relation, attn_weights_tail):
        epsilon = torch.exp(self.log_epsilon)
        if epsilon < self.reset_threshold:
            with torch.no_grad():
                self.log_epsilon.fill_(math.log(self.init_epsilon))
            epsilon = torch.exp(self.log_epsilon)
        multi_result = torch.mul(torch.mul(attn_weights_head, attn_weights_relation), attn_weights_tail) + epsilon
        res = torch.pow(multi_result, 1.0/3)
        return res
@dataclass
class ExpendModelOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    subgraph_embedding: torch.FloatTensor = None


@dataclass
class ExpendSeq2SeqModelOutput(Seq2SeqModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    subgraph_embedding: torch.FloatTensor = None


@dataclass
class ExpendSeq2SeqLMOutput(Seq2SeqLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    subgraph_embedding: torch.FloatTensor = None


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        # 适当放大几何注意力机制的缩放因子，防止出现梯度爆炸的情况
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        ###########################################################################魔改注意力投影矩阵
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        # with open("hidden_states.pkl", 'wb') as f:
        #     pickle.dump(hidden_states, f)

        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:  # decoder encod attention
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:  # decoder self attention
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:  # encoder
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        # with open('key_states_prefix', 'wb') as f:
        #     pickle.dump(proj_shape, f)
        #     pickle.dump(key_states, f)
        # exit()
        # key_states = key_states.reshape(*proj_shape)
        # value_states = value_states.reshape(*proj_shape)
        key_states = key_states.contiguous().view(*proj_shape)
        value_states = value_states.contiguous().view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BartAttention_GraphAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # self.num_heads = num_heads
        self.num_heads = 1
        self.dropout = dropout
        # self.head_dim = embed_dim // num_heads
        self.head_dim = embed_dim
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5  # 这个就是开的根号，做分母的
        self.is_decoder = is_decoder
        ######################################## 这里的query是输入的hidden_states，key和value都是知识子图的嵌入
        self.k_proj_head = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.v_proj_head = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.k_proj_relation = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.v_proj_relation = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.k_proj_tail = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.v_proj_tail = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        ########################################

        """
        原论文里面query也是有三个投影矩阵的
        """
        self.q_proj_head = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj_relation = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj_tail = nn.Linear(embed_dim, embed_dim, bias=bias)
        #############################################################################
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.get_geometricMean = AdaptiveEpsilonGeometricMean()
    def log_sum_exp(self,x, dim=-1):
        max_x = torch.max(x, dim=dim, keepdim=True)[0]
        return max_x + torch.log(torch.sum(torch.exp(x - max_x), dim=dim, keepdim=True))

    def stable_softmax(self,x, dim=-1):
        return torch.exp(x - self.log_sum_exp(x, dim=dim))
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            key_value_states_head: Optional[torch.Tensor] = None,
            key_value_states_relation: Optional[torch.Tensor] = None,
            key_value_states_tail: Optional[torch.Tensor] = None,
            value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states_head is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        # with open("hidden_states.pkl", 'wb') as f:
        #     pickle.dump(hidden_states, f)
        # print("初始输入的query_states", hidden_states)
        query_states_head = self.q_proj_head(hidden_states) * self.scaling
        query_states_relation = self.q_proj_relation(hidden_states) * self.scaling
        query_states_tail = self.q_proj_tail(hidden_states) * self.scaling
        # print("投影映射后的query_states", query_states)

        # get key, value proj
        if is_cross_attention and past_key_value is not None:  # decoder encod attention
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            #######################################################################################################
            key_states_head = self._shape(self.k_proj_head(key_value_states_head), -1, bsz)
            # value_states_head = self._shape(self.v_proj(key_value_states_head), -1, bsz)

            key_states_relation = self._shape(self.k_proj_relation(key_value_states_relation), -1, bsz)
            # value_states_relation = self._shape(self.v_proj_relation(key_value_states_relation), -1, bsz)

            key_states_tail = self._shape(self.k_proj_tail(key_value_states_tail), -1, bsz)
            # value_states_tail = self._shape(self.v_proj_tail(key_value_states_tail), -1, bsz)

            value_states = self._shape(self.v_proj(value_states), -1, bsz)
            #######################################################################################################
        elif past_key_value is not None:  # decoder self attention
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:  # encoder
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        ################################################################
        query_states_head = self._shape(query_states_head, tgt_len, bsz).view(*proj_shape)
        query_states_relation = self._shape(query_states_relation, tgt_len, bsz).view(*proj_shape)
        query_states_tail = self._shape(query_states_tail, tgt_len, bsz).view(*proj_shape)
        ################################################################
        # with open('key_states_prefix', 'wb') as f:
        #     pickle.dump(proj_shape, f)
        #     pickle.dump(key_states, f)
        # exit()
        # key_states = key_states.reshape(*proj_shape)
        # value_states = value_states.reshape(*proj_shape)

        #########################################################################################
        key_states_head = key_states_head.contiguous().view(*proj_shape)
        # value_states_head = value_states_head.contiguous().view(*proj_shape)

        key_states_relation = key_states_relation.contiguous().view(*proj_shape)
        # value_states_relation = value_states_relation.contiguous().view(*proj_shape)

        key_states_tail = key_states_tail.contiguous().view(*proj_shape)
        # value_states_tail = value_states_tail.contiguous().view(*proj_shape)

        value_states = value_states.contiguous().view(*proj_shape)
        src_len = key_states_head.size(1)

        #########################################################################################
        attn_weights_head = torch.bmm(query_states_head, key_states_head.transpose(1, 2))
        # print('计算出来的attn_weights_head', attn_weights_head)
        # if torch.isnan(attn_weights_head).any():
        #     print('___________________________________')
        #     print('query_states', query_states)
        #     print('key_states_head', key_states_head.transpose(1, 2))
        attn_weights_relation = torch.bmm(query_states_relation, key_states_relation.transpose(1, 2))
        attn_weights_tail = torch.bmm(query_states_tail, key_states_tail.transpose(1, 2))

        if attn_weights_head.size() != (bsz * self.num_heads, tgt_len, src_len) or attn_weights_relation.size() != (
        bsz * self.num_heads, tgt_len, src_len) or attn_weights_tail.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights_head.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            #########################################################################################
            # print('attention_mask', attention_mask)
            attn_weights_head = attn_weights_head.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights_head = attn_weights_head.view(bsz * self.num_heads, tgt_len, src_len)

            attn_weights_relation = attn_weights_relation.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights_relation = attn_weights_relation.view(bsz * self.num_heads, tgt_len, src_len)

            attn_weights_tail = attn_weights_tail.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights_tail = attn_weights_tail.view(bsz * self.num_heads, tgt_len, src_len)
        # print('归一化之前的attn_weights_head', attn_weights_head)
        attn_weights_head = self.stable_softmax(attn_weights_head, dim=-1)
        # print("attn_weights_head", attn_weights_head)
        attn_weights_relation = self.stable_softmax(attn_weights_relation, dim=-1)
        # print("attn_weights_relation", attn_weights_relation)
        attn_weights_tail = self.stable_softmax(attn_weights_tail, dim=-1)
        # print("attn_weights_tail", attn_weights_tail)
        ######################################################################################### 得到三个注意力得分，计算几何均值
        """
        使用自定义的几何均值计算方法进行几何均值的计算
        """
        # attn_weights = torch.mul(torch.mul(attn_weights_head, attn_weights_relation), attn_weights_tail)
        # epsilon = 1e-5
        # attn_weights = attn_weights + epsilon
        # attn_weights = (torch.log(attn_weights_head) + torch.log(attn_weights_relation) + torch.log(attn_weights_tail)) / 3
        # attn_weights = torch.exp(attn_weights)
        # print('attn_weights三个点乘以后的结果', attn_weights)
        #这三个东西点乘之后的数据量级过小，加一个小的正则化项
        ##################################### 三个注意力机制的乘积得到的结果，数据的数量级过小，需要进行数据放大
        # attn_weights = attn_weights * self.scaling2
        # attn_weights = torch.pow(attn_weights, 1.0 / 3)
        attn_weights = self.get_geometricMean(attn_weights_head, attn_weights_relation, attn_weights_tail)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # 先用算数平均进行测试 这里没有问题，使用几何平均便会报错
        # print("attn_weights!!!!!!!!!!!!!!!!!!!!!!!!", attn_weights)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)#相当于经过一个全连接层
        # print('attn_output', attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


######################################################################################
class MemoryAttention(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 嵌入的维度等于模型本身的维度
        self.self_attn = BartAttention(  # 使用bart自身的注意力函数
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.r_proj = nn.Linear(3, 1, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            memory_bank_embeds: torch.Tensor,
            memory_bank_attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
    ):
        # 2*3*768 -> 2*768 2代表着每条样本对应的知识库中的两个三元组向量，
        # 一开始的维度顺序是样本列表、知识库列表（2个）、知识向量矩阵（3个）、hrt向量（768）？换完顺序为a*2*768*3 #相当于二维矩阵的转置
        memory_bank_embeds = self.r_proj(memory_bank_embeds.permute(0, 1, 3, 2)).squeeze()
        # 相当于论文中的Kh矩阵
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=memory_bank_embeds,
            attention_mask=memory_bank_attention_mask,  # 通过attention mask来标记每一位是否关注
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        return hidden_states  # 此处计算出来的隐状态对应于全局知识基础的内容


######################################################################################
class GraphAttention(nn.Module):
    # 几何注意力机制实现
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 嵌入的维度等于模型本身的维度
        self.self_attn = BartAttention_GraphAttention(  # 使用bart自身的注意力函数
            embed_dim = self.embed_dim,
            num_heads = 1,#固定为单个头的
            dropout=config.attention_dropout,
        )
        self.r_proj = nn.Linear(3, 1, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            memory_bank_embeds: torch.Tensor,
            memory_bank_embeds_head: torch.Tensor,
            memory_bank_embeds_relation: torch.Tensor,
            memory_bank_embeds_tail: torch.Tensor,
            memory_bank_attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
    ):
        # 2*3*768 -> 2*768 2代表着每条样本对应的知识库中的两个三元组向量，
        # 一开始的维度顺序是样本列表、知识库列表（2个）、知识向量矩阵（3个）、hrt向量（768）？换完顺序为a*2*768*3 #相当于二维矩阵的转置
        # 相当于论文中的Kh矩阵
        # print('memory_bank_attention_mask', memory_bank_attention_mask)
        memory_bank_embeds = self.r_proj(memory_bank_embeds.permute(0, 1, 3, 2)).squeeze()

        hidden_states, graph_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            value_states=memory_bank_embeds,
            key_value_states_head=memory_bank_embeds_head,
            key_value_states_relation=memory_bank_embeds_relation,
            key_value_states_tail=memory_bank_embeds_tail,
            attention_mask=memory_bank_attention_mask,  # 通过attention mask来标记每一位是否关注
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        return hidden_states  # 此处计算出来的隐状态对应于全局知识基础的内容


######################################################################################
###################################################################################### 门控机制的实现
class GatingMechanism(nn.Module):
    def __init__(self, d_model):
        super(GatingMechanism, self).__init__()
        # 定义门控机制的权重矩阵 Wo_K_f
        self.Wo_K_f = nn.Parameter(torch.randn(1, d_model))
        self.sigmoid = nn.Sigmoid()

    def forward(self, Ho_f, Ho_K_f):
        # 计算门控信号p
        p = self.sigmoid(torch.matmul(Ho_K_f, self.Wo_K_f.t()))
        # 计算最终输出Ho
        Ho = p * Ho_K_f + (1 - p) * Ho_f
        return Ho
######################################################################################

class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (:obj:`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartPretrainedModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class PretrainedBartModel(BartPretrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPretrainedModel` instead.",
            FutureWarning,
        )


BART_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BART_GENERATION_EXAMPLE = r"""
    Summarization example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        >>> print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

    Mask filling example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        >>> TXT = "My friends are <mask> but they eat too many carbs."

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are decoder input IDs? <../glossary.html#decoder-input-ids>`__

            Bart uses the :obj:`eos_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            For translation and summarization training, :obj:`decoder_input_ids` should be provided. If no
            :obj:`decoder_input_ids` is provided, the model will create this tensor by shifting the :obj:`input_ids` to
            the right for denoising pre-training following the paper.
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read :func:`modeling_bart._prepare_decoder_inputs` and
            modify to your needs. See diagram 1 in `the paper <https://arxiv.org/abs/1910.13461>`__ for more
            information on the default strategy.
        head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in ``[0,
            1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class BartEncoder(BartPretrainedModel):  # 改写Bart模型的编码器部分，具体来说就是对输入n层transformer的起始序列做文章
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None,
                 embed_entitys_relations: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.embedding_dim = embed_dim
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        ################################################################
        # 这段为作者新增的针对KG融合的代码
        self.mode = config.mode  # 设定了一个模式？
        self.memory_bank_mode = config.memory_bank_mode  # 知识库模式：实体提及每个分词都替换还是仅替换第一个词
        self.use_kg_embedding = config.use_kg_embedding  # 是否使用kg嵌入？
        self.use_memory_bank = config.use_memory_bank  # 是否使用知识库

        self.embed_entitys_relations = embed_entitys_relations  # 编码器中也增加KG嵌入矩阵
        self.embed_entitys_relations.weight.requires_grad = False  # 这个嵌入矩阵不需要梯度，因为没有更新

        original_graph_emb_dim = self.embed_entitys_relations.embedding_dim
        if original_graph_emb_dim == embed_dim:  # KG嵌入的维度和Bart模型本身对文本嵌入的维度相同
            self.embed_proj = None  # 则不用使用投影矩阵
        else:
            # 否则增加一个投影层，将KG嵌入映射到文本空间
            # print("original_graph_emb_dim: ", original_graph_emb_dim, "and embed_dim: ", embed_dim)
            self.embed_proj = nn.Linear(original_graph_emb_dim, embed_dim, bias=False)  # 直接做线性映射
            # print('KG嵌入映射矩阵的设备', self.embed_proj.weight.device)

        if self.mode == 'last_one':
            self.layernorm_last = nn.LayerNorm(embed_dim)

        if self.use_memory_bank:  # 使用知识库的话，计算全局知识基础时同样需要判断是否需要进行映射
            if original_graph_emb_dim == embed_dim:
                self.embed_proj2 = None
                self.embed_proj_head = None
                self.embed_proj_relation = None
                self.embed_proj_tail = None
            else:
                self.embed_proj2 = nn.Linear(original_graph_emb_dim, embed_dim, bias=False)
                ############################################################################################## 参考2024ACL的论文思路
                self.embed_proj_head = nn.Linear(original_graph_emb_dim, embed_dim, bias=False)
                self.embed_proj_relation = nn.Linear(original_graph_emb_dim, embed_dim, bias=False)
                self.embed_proj_tail = nn.Linear(original_graph_emb_dim, embed_dim, bias=False)
                ##############################################################################################
                # print('子图全局嵌入映射矩阵的设备', self.embed_proj2.weight.device)
            self.get_memory_attention = MemoryAttention(config)  # 全局知识基础的注意力机制
            ############################################################################## 参考2024ACL的论文思路，分别进行头实体、关系、尾实体的注意力机制计算
            # self.get_memory_attention_head = GraphAttention(config)
            # self.get_memory_attention_relation = GraphAttention(config)
            # self.get_memory_attention_tail = GraphAttention(config)
            self.get_memory_attention_geometric = GraphAttention(config)
            self.get_gate = GatingMechanism(d_model = embed_dim)
            ##############################################################################
        self.encoder_layers = config.encoder_layers
        ################################################################

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            input_entity_relation_ids=None,
            neg_pos_label=None,
            memory_bank=None,  # memory bank只是保留了entity和relation的id，那么id是怎么转化成嵌入向量的？
            memory_bank_attention_mask=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        #################################################################################
        # 使用KG嵌入
        if self.use_kg_embedding:  # 使用局部知识基础
            entity_relation_embeds = self.embed_entitys_relations(input_entity_relation_ids)  # 获取所有的对应嵌入，pad对应全零向量
            # 输入idx向量，获取局部知识基础的w
            if self.embed_proj:
                # [bsz, seq_len, orginal_dim] -> [bsz, seq_len, embed_dim]
                entity_relation_embeds = self.embed_proj(entity_relation_embeds)  # 将局部知识基础映射到文本域

        if self.use_memory_bank:  # 全局知识基础
            if memory_bank_attention_mask is not None:
                if self.memory_bank_mode == 'all':
                    # [bsz, seq_len=2] -> [bsz, 1, tgt_seq_len, src_seq_len=2]
                    # print('执行这个！！！！！！！！！！！！')
                    # 获取注意力机制掩码
                    memory_bank_attention_mask = _expand_mask(memory_bank_attention_mask, inputs_embeds.dtype,
                                                              tgt_len=attention_mask.size()[-1])
                elif self.memory_bank_mode == 'entity':
                    # [bsz, seq_len=2] -> [bsz, 1, tgt_seq_len, src_seq_len=2]
                    # [批量大小，[知识库的注意力掩码，长度为2的向量]]->[bsz,1,目标序列长度，源序列长度]
                    memory_bank_attention_mask = entity_expand_attention_mask(memory_bank_attention_mask,
                                                                              input_entity_relation_ids, attention_mask,
                                                                              inputs_embeds.dtype)
                else:
                    print("memory_bank_mode must be all or entity!!")
                    assert False
            ################################################################################################### 仿照2024ACL论文的实现
            # print("memory_bank_shape", memory_bank.shape) #memory_bank的真实形状第一维度是batch_size,第二维度才是
            memory_bank_head = memory_bank[:, :, 0]
            # print("memory_bank_head", memory_bank_head)
            memory_bank_relation = memory_bank[:, :, 1]
            # print("memory_bank_head", memory_bank_relation)
            memory_bank_tail = memory_bank[:, :, 2]
            # print("memory_bank_head", memory_bank_tail)
            memory_bank_head = self.embed_entitys_relations(memory_bank_head)
            memory_bank_relation = self.embed_entitys_relations(memory_bank_relation)
            memory_bank_tail = self.embed_entitys_relations(memory_bank_tail)
            # print("memory_bank_head shape:", memory_bank_head.shape)
            if self.embed_proj_head:
                memory_bank_head = self.embed_proj_head(memory_bank_head)
            if self.embed_proj_relation:
                memory_bank_relation = self.embed_proj_relation(memory_bank_relation)
            if self.embed_proj_tail:
                memory_bank_tail = self.embed_proj_tail(memory_bank_tail)
            ###################################################################################################
            memory_bank_embeds = self.embed_entitys_relations(memory_bank)  # 知识库的嵌入获取，获取论文中的Vi拼接的矩阵
            if self.embed_proj2:
                memory_bank_embeds = self.embed_proj2(memory_bank_embeds)  # 将知识库映射到文本域，相当于映射操作M(.)

        if self.mode == 'input':
            if self.use_memory_bank:
                if torch.isnan(hidden_states).any():
                    print("hidden_states is nan")
                    print('hidden_states', hidden_states)
                memory_attention_geometric = self.get_memory_attention_geometric(
                    hidden_states=hidden_states,  # query 这个hidden states是否是输入n层transformer的text embedding？
                    memory_bank_embeds=memory_bank_embeds,  # key and value
                    memory_bank_embeds_head=memory_bank_head,
                    memory_bank_embeds_relation=memory_bank_relation,
                    memory_bank_embeds_tail=memory_bank_tail,
                    memory_bank_attention_mask=memory_bank_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    #### 这个地方的错误写法怎么改？ 头掩码矩阵为空,不会报错，因为bart attention中有判断为空的逻辑
                    output_attentions=output_attentions,
                )
                # memory_attention = self.get_memory_attention(
                #     hidden_states=hidden_states,#query 这个hidden states是否是输入n层transformer的text embedding？
                #     memory_bank_embeds=memory_bank_embeds,#key and value
                #     memory_bank_attention_mask=memory_bank_attention_mask,
                #     layer_head_mask=(head_mask[idx] if head_mask is not None else None),#### 这个地方的错误写法怎么改？ 头掩码矩阵为空,不会报错，因为bart attention中有判断为空的逻辑
                #     output_attentions=output_attentions,)
                ################################################################## 参考2024ACL论文的实现
                # memory_attention_head = self.get_memory_attention_head(
                #     hidden_states=hidden_states,#query 这个hidden states是否是输入n层transformer的text embedding？
                #     graph_embeds=memory_bank_head,#key and value
                #     memory_bank_attention_mask=memory_bank_attention_mask,
                #     layer_head_mask=(head_mask[idx] if head_mask is not None else None),#### 这个地方的错误写法怎么改？ 头掩码矩阵为空,不会报错，因为bart attention中有判断为空的逻辑
                # )
                # print("memory_attention_head shape:", memory_attention_head.shape)
                # print("memory_attention_head", memory_attention_head)
                # memory_attention_relation = self.get_memory_attention_relation(
                #     hidden_states=hidden_states,#query 这个hidden states是否是输入n层transformer的text embedding？
                #     graph_embeds=memory_bank_relation,#key and value
                #     memory_bank_attention_mask=memory_bank_attention_mask,
                #     layer_head_mask=(head_mask[idx] if head_mask is not None else None),#### 这个地方的错误写法怎么改？ 头掩码矩阵为空,不会报错，因为bart attention中有判断为空的逻辑
                # )
                # print("memory_attention_relation shape:", memory_attention_relation.shape)
                # print("memory_attention_relation", memory_attention_relation)
                # memory_attention_tail = self.get_memory_attention_tail(
                #     hidden_states=hidden_states,#query 这个hidden states是否是输入n层transformer的text embedding？
                #     graph_embeds=memory_bank_tail,#key and value
                #     memory_bank_attention_mask=memory_bank_attention_mask,
                #     layer_head_mask=(head_mask[idx] if head_mask is not None else None),#### 这个地方的错误写法怎么改？ 头掩码矩阵为空,不会报错，因为bart attention中有判断为空的逻辑
                # )
                # print("memory_attention_tail shape:", memory_attention_tail.shape)
                # print("memory_attention_tail", memory_attention_tail)
                # # 计算三个嵌入的哈达玛积并开立方根
                # memory_attention_Geometric = torch.mul(torch.mul(memory_attention_head, memory_attention_relation), memory_attention_tail)
                # memory_attention_Geometric = torch.pow(memory_attention_Geometric, 1.0/3)
                # print("Geometric shape:", memory_attention_Geometric.shape)
                # print("Geometric:", memory_attention_Geometric)

            if self.use_kg_embedding and self.use_memory_bank:  # 使用局部知识基础和全局知识基础
                hidden_states = hidden_states + entity_relation_embeds + memory_attention_geometric
            elif self.use_kg_embedding:
                hidden_states = hidden_states + entity_relation_embeds
            elif self.use_memory_bank:
                hidden_states = hidden_states + memory_attention_geometric
        #################################################################################
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                ############################################################################################################
                # if (self.mode == 'last_half' and idx >= self.encoder_layers//2) or (self.mode == 'interval' and (idx+1)%2):
                if (self.mode == 'last_one' and idx == (self.encoder_layers - 1)):
                    if self.use_memory_bank:
                        # memory_attention需要每次更新吗
                        # memory_attention = self.get_memory_attention(
                        #     hidden_states=hidden_states,#query
                        #     memory_bank_embeds=memory_bank_embeds,#key and value
                        #     memory_bank_attention_mask=memory_bank_attention_mask,
                        #     layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        #     output_attentions=output_attentions,)
                        memory_attention_geometric = self.get_memory_attention_geometric(
                            hidden_states=hidden_states,  # query 这个hidden states是否是输入n层transformer的text embedding？
                            memory_bank_embeds=memory_bank_embeds,  # key and value
                            memory_bank_embeds_head=memory_bank_head,
                            memory_bank_embeds_relation=memory_bank_relation,
                            memory_bank_embeds_tail=memory_bank_tail,
                            memory_bank_attention_mask=memory_bank_attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            #### 这个地方的错误写法怎么改？ 头掩码矩阵为空,不会报错，因为bart attention中有判断为空的逻辑
                            output_attentions=output_attentions,
                        )

                    if self.use_kg_embedding and self.use_memory_bank:
                        hidden_states = hidden_states + entity_relation_embeds + memory_attention_geometric
                    elif self.use_kg_embedding:
                        hidden_states = hidden_states + entity_relation_embeds
                    elif self.use_memory_bank:
                        hidden_states = hidden_states + memory_attention_geometric

                    hidden_states = self.layernorm_last(hidden_states)
                    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
                ############################################################################################################
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        ############################################################################################################## 计算一个子图三元组的嵌入向量，用来进行对比学习
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 如果GPU设备可以，则用GPU；否则用CPU设备
        self.r_proj_subgraph = nn.Linear(3, 1, bias=False, device=device)  # 定义一个用于将三元组矩阵映射为单个向量的矩阵
        # print('映射矩阵的设备',self.r_proj_subgraph.weight.device)
        subgraph_embeds = memory_bank_embeds
        subgraph_embeds = subgraph_embeds.to(device)
        # permute: [B, n_triple, emb_dim, 3] → [B, n_triple, 3, emb_dim]
        # Linear(3→1): [B, n_triple, emb_dim, 1]
        # squeeze(-1): [B, n_triple, emb_dim]（只 squeeze 最后一维，避免 n_triple=1 时误 squeeze）
        subgraph_embeds = self.r_proj_subgraph(
            subgraph_embeds.permute(0, 1, 3, 2)).squeeze(-1)  # [B, n_triple, emb_dim]
        # 对三元组维度求均值，得到子图整体表示 [B, emb_dim]
        subgraph_embeds = torch.mean(subgraph_embeds, dim=1)
        # print('子图嵌入的形状是：',subgraph_embeds.shape)
        ##############################################################################################################

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions, subgraph_embeds] if v is not None)
        # return BaseModelOutput(
        #     last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        # )

        # 在此处多返回一个知识库的向量，方便后续进行对比学习
        return ExpendModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            subgraph_embedding=subgraph_embeds
        )


class BartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        # with open('past_key_values', 'wb') as f:
        #     pickle.dump(past_key_values, f)
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig, entity_relation_weight):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)  # 创建嵌入曾
        ####################################################################
        # 增加KG嵌入层，将实体和关系的嵌入向量进行保存
        original_graph_emb_dim = entity_relation_weight.shape[1]
        self.shared_entity_relation = nn.Embedding(entity_relation_weight.shape[0], original_graph_emb_dim,
                                                   padding_idx=padding_idx)
        # shape[0]是KG嵌入的总数
        self.shared_entity_relation.weight.data.copy_(torch.from_numpy(entity_relation_weight))  # 为模型增加KG嵌入
        self.shared_entity_relation.weight.requires_grad = False  # KG嵌入不增加梯度，因为没有涉及计算
        ####################################################################

        self.encoder = BartEncoder(config, self.shared, self.shared_entity_relation)  # 向编码器中传入KG嵌入
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids,
            input_entity_relation_ids,  # 输入的实体关系提及位置标注
            memory_bank,  # 知识库
            neg_pos_label=None,
            memory_bank_attention_mask=None,  # 知识库注意力掩码
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:  # 编码器还没有输出
            encoder_outputs = self.encoder(  # 向编码器中输入实体关系提及位置、知识库以及KG嵌入矩阵
                input_ids=input_ids,
                input_entity_relation_ids=input_entity_relation_ids,
                neg_pos_label=neg_pos_label,
                memory_bank=memory_bank,
                memory_bank_attention_mask=memory_bank_attention_mask,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return ExpendSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            subgraph_embedding=encoder_outputs.subgraph_embedding  # 增加返回一个子图嵌入，用于后续计算对比学习的损失
        )


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)
class BartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, entity_relation_weight):
        super().__init__(config)
        self.model = BartModel(config, entity_relation_weight)  # 往bart-model里多传入了实体关系权重？
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # print("self.model.shared.num_embeddings", self.model.shared.num_embeddings)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)  # 语言模型的输出头，将hidden state
        # 权重初始化
        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @replace_return_docstrings(output_type=ExpendSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids=None,
            input_entity_relation_ids=None,
            #################################
            neg_pos_label=None,
            #################################
            memory_bank=None,
            memory_bank_attention_mask=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            neg_pos_label=neg_pos_label,
            input_entity_relation_ids=input_entity_relation_ids,
            memory_bank=memory_bank,
            memory_bank_attention_mask=memory_bank_attention_mask,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # print("self.config.vocab_size", self.config.vocab_size, lm_logits.size(), labels.size())
            # self.config.vocab_size 50270 torch.Size([1, 64, 50270]) torch.Size([1, 64])
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return ExpendSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            subgraph_embedding=outputs.subgraph_embedding
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
