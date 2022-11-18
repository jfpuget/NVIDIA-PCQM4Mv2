# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. Copyright (c) Microsoft Corporation. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import math
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.categorical import Categorical


from .kpgt_pretext import KPGTPretext
from .graphormer_layers import NodePositionHead
from .graphormer_layers import GraphAttnBias
from .graphormer_layers import GraphNodeFeature
from .graphormer_layers import Structural3DFeatures
from .multihead_attention import MultiheadAttention
from .multihead_attention import PartialMultiheadAttention
from .nn_utils import get_activation_fn, RandomTP


def init_graphormer_params(module, std=0.02):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, PartialMultiheadAttention):
        nn.init.xavier_uniform_(module.qkv_proj.weight, gain=1 / math.sqrt(2))
    if isinstance(module, MultiheadAttention):
        nn.init.xavier_uniform_(module.out_proj.weight)
        if module.out_proj.bias is not None:
            nn.init.constant_(module.out_proj.bias, 0.0)


@torch.jit.script
def residual_dropout_layernorm(x, residual, dropout: float, drop_path: float,
                               training: bool,
                               layernorm_shape: Tuple[int],
                               layernorm_weight: Tensor,
                               layernorm_bias: Tensor):
    mask = torch.rand((1, x.shape[1], 1), device=x.device) >= drop_path * float(training)
    x = x / (1 - drop_path * float(training)) * mask
    y = residual + x
    y = F.dropout(y, dropout, training)
    y = F.layer_norm(y.float(), layernorm_shape, layernorm_weight, layernorm_bias)
    return y.type_as(x)


@torch.jit.script
def dropout_residual_layernorm(x, residual, dropout: float, drop_path: float,
                               training: bool,
                               layernorm_shape: Tuple[int],
                               layernorm_weight: Tensor,
                               layernorm_bias: Tensor):
    mask = torch.rand((1, x.shape[1], 1), device=x.device) >= drop_path * float(training)
    x = x / (1 - drop_path * float(training)) * mask
    y = F.dropout(x, dropout, training)
    y = residual + y
    y = F.layer_norm(y.float(), layernorm_shape, layernorm_weight, layernorm_bias)
    return y.type_as(x)


class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        drop_path_prob: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.drop_path_prob = drop_path_prob

        self.activation_fn = activation_fn
        self.self_attention = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            attention_dropout,
        )

        # layer norm associated with the self attention layer
        self.self_attention_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.ffn_in = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.act_dropout = torch.jit.script(nn.Sequential(
            get_activation_fn(activation_fn),
            nn.Dropout(activation_dropout),
        ))
        self.ffn_out = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
    ):
        # x: T x B x C
        residual = x
        x = self.self_attention(
            x,
            attn_bias=self_attn_bias,
        )
        x = residual_dropout_layernorm(
            x, residual,
            self.dropout,
            self.drop_path_prob,
            self.training,
            self.self_attention_layer_norm.normalized_shape,
            self.self_attention_layer_norm.weight,
            self.self_attention_layer_norm.bias
        )
        residual = x
        x = self.ffn_in(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = self.act_dropout(x)
        x = self.ffn_out(x)
        x = dropout_residual_layernorm(
            x, residual,
            self.dropout,
            self.drop_path_prob,
            self.training,
            self.final_layer_norm.normalized_shape,
            self.final_layer_norm.weight,
            self.final_layer_norm.bias
        )
        return x


class GraphormerGraphEncoder(nn.Module):
    def __init__(
        self,
        num_atom_embeddings: int,
        max_in_degree: int,
        max_out_degree: int,
        num_edge_embeddings: int,
        num_spatial_embeddings: int,
        num_edge_dist_embeddings: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        encoder_normalize_before: bool = True,
        apply_graphormer_init: bool = True,
        activation_fn: str = "gelu",
        channel_prob_2d_only: float = 0.2,
        channel_prob_3d_only: float = 0.6,
        channel_prob_2d_3d: float = 0.2,
        num_kernels: int = 128,
        drop_path_prob: float = 0.1,
        node_pos_head: bool = False,
        node_pos_attention_dropout: float = 0.1,
        position_noise: float = 0.0,
        random_tp: bool = False,
        kpgt: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.position_noise = position_noise
        self.apply_graphormer_init = apply_graphormer_init

        if random_tp:
            self.channels_distribution = RandomTP(channel_prob_2d_only, channel_prob_2d_3d, channel_prob_3d_only)
        else:
            self.channels_distribution = Categorical(
                probs=torch.tensor([channel_prob_2d_only, channel_prob_2d_3d, channel_prob_3d_only])
            )

        self.dropout = nn.Dropout(dropout)
        self.graph_node_feature = GraphNodeFeature(
            num_atom_embeddings=num_atom_embeddings,
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )
        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_edge_embeddings=num_edge_embeddings,
            num_spatial_embeddings=num_spatial_embeddings,
            num_edge_dist_embeddings=num_edge_dist_embeddings,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            n_layers=num_encoder_layers,
        )

        if channel_prob_3d_only + channel_prob_2d_3d > 0:
            self.channel_3d = Structural3DFeatures(
                num_kernels=num_kernels,
                num_embeddings=num_atom_embeddings * 2,
                embedding_dim=embedding_dim,
                num_heads=num_attention_heads,
                activation_fn=activation_fn,
            )

        self.kpgt_head = None
        if kpgt:
            self.kpgt_head = KPGTPretext(
                model_dim=embedding_dim,
            )
 
        if node_pos_head:
            self.node_pos_head = NodePositionHead(
                embed_dim=embedding_dim,
                num_heads=num_attention_heads,
                dropout=node_pos_attention_dropout,
            )

        self.emb_layer_norm = None
        if encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.encoder_layers = nn.ModuleList([
            GraphormerGraphEncoderLayer(
                embedding_dim=embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                drop_path_prob=drop_path_prob * i / (num_encoder_layers - 1),
            ) for i in range(num_encoder_layers)
        ])

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

    def forward(self, batched_data) -> Dict[str, Tensor]:
        data_x = batched_data['x']
        padding_mask = (data_x[:, :, 0] == 0)

        # Transformer-M structural channels
        edge_3d_bias, atom_3d_feats, delta_pos = None, None, None
        channels_2d_mask, channels_3d_mask = None, None
        pos_noise = None
        out = {}
        if self.training and 'node_pos' in batched_data and 'node_type_pair' in batched_data:
            channels_idx = self.channels_distribution.sample((data_x.shape[0],))
            channels_2d_mask, channels_3d_mask = channels_idx <= 1, channels_idx >= 1
            out["channels_3d_mask"] = channels_3d_mask
            out["channels_2d_mask"] = channels_2d_mask
            if channels_3d_mask.any():
                node_pos = batched_data['node_pos']
                pos_noise = torch.randn_like(node_pos) * self.position_noise
                node_pos = node_pos + pos_noise
                out["pos_noise"] = pos_noise
                edge_3d_bias, atom_3d_feats, delta_pos = self.channel_3d(
                    node_pos[channels_3d_mask],
                    batched_data['node_type_pair'][channels_3d_mask],
                    padding_mask[channels_3d_mask]
                )
        else:
            channels_2d_mask = ...

        x = self.graph_node_feature(batched_data)

        if atom_3d_feats is not None:
            x[channels_3d_mask, 1:] += 0.01 * atom_3d_feats

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout(x)
        x = x.transpose(0, 1)  # sequence first

        attn_bias = batched_data['attn_bias'].unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        if attn_bias[channels_2d_mask].numel():
            attn_bias[channels_2d_mask] += self.graph_attn_bias(
                attn_bias[channels_2d_mask],
                batched_data['spatial_pos'][channels_2d_mask],
                batched_data['edge_input'][channels_2d_mask],
                batched_data['attn_edge_type'][channels_2d_mask]
            )

        if edge_3d_bias is not None:
            attn_bias[channels_3d_mask, :, 1:, 1:] += edge_3d_bias

        if torch.is_autocast_enabled():
            attn_bias = attn_bias.half()

        for layer in self.encoder_layers:
            x = layer(
                x,
                self_attn_bias=attn_bias,
            )
        out["pred"] = x
        # - kpgt
        kpgt_out = None
        if self.kpgt_head is not None:
            kpgt_out = self.kpgt_head(
                graph_reps=x[0, :, :],
                indices=batched_data['idx'])
            out["kpgt"] = kpgt_out

        node_pos_head = None
        if self.training and delta_pos is not None and hasattr(self, 'node_pos_head'):
            node_pos_head = self.node_pos_head(
                x[1:, channels_3d_mask], attn_bias[channels_3d_mask, :, 1:, 1:], delta_pos
            )
            out["node_pos_head"] = node_pos_head
        
        return out
