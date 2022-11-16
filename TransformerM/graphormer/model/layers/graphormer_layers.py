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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphormer.model.layers.multihead_attention import PartialMultiheadAttention
from graphormer.model.layers.multihead_attention import softmax_dropout
from graphormer.model.layers.nn_utils import gaussian_basis_function
from graphormer.model.layers.nn_utils import get_activation_fn


def init_params(module: nn.Module, n_layers: int):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding) or isinstance(module, nn.EmbeddingBag):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GBFEncoder(nn.Module):
    def __init__(self, num_kernels=256, num_embeddings=256):
        super().__init__()
        self.means = nn.Parameter(torch.empty(num_kernels).uniform_(0, 1))
        self.stds = nn.Parameter(torch.empty(num_kernels).uniform_(0, 1))
        self.scale = nn.Embedding(num_embeddings, 1, padding_idx=0)
        self.bias = nn.Embedding(num_embeddings, 1, padding_idx=0)
        nn.init.constant_(self.scale.weight, 1)
        nn.init.constant_(self.bias.weight, 0)

    def forward(self, d, extra):
        scale = self.scale(extra).sum(dim=-2)
        bias = self.bias(extra).sum(dim=-2)
        d = scale * d.unsqueeze(-1) + bias
        x = gaussian_basis_function(d, self.means[None, None, None, :], self.stds[None, None, None, :])
        return x


class FastLargeInputEmbeddingBagFunction(torch.autograd.Function):
    """
    Solves the slowness of native embedding/embeddingbag backward pass
    when the input is large and only uses a few embeddings,
    creating many collisions in the backward accumulations.
    """

    @staticmethod
    def forward(ctx, idx, weights):
        result = F.embedding_bag(idx, weights, padding_idx=0)
        ctx.save_for_backward(idx)
        ctx.weights_shape = weights.shape
        return result

    @staticmethod
    def backward(ctx, grad_output):
        idx, = ctx.saved_tensors
        idx_multihot = torch.zeros(idx.shape[0], ctx.weights_shape[0], dtype=grad_output.dtype,
                                   device=grad_output.device)
        idx_multihot.scatter_(-1, idx, 1.0)
        out = idx_multihot.T[1:] @ grad_output.flatten(0, -2)
        out = out / idx.shape[-1]
        out = F.pad(out, (0, 0, 1, 0))
        return None, out.view(*ctx.weights_shape)


class FastLargeInputEmbeddingBag(torch.nn.EmbeddingBag):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim, padding_idx=0)

    def forward(self, idx):
        return FastLargeInputEmbeddingBagFunction.apply(idx, self.weight)


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self,
        num_atom_embeddings: int,
        max_in_degree: int,
        max_out_degree: int,
        hidden_dim: int,
        n_layers: int,
    ):
        super(GraphNodeFeature, self).__init__()
        self.atom_encoder = nn.Embedding(num_atom_embeddings + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(max_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(max_out_degree, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x = batched_data["x"]
        in_degree = batched_data["in_degree"]
        out_degree = batched_data["out_degree"]
        n_graphs = x.shape[0]

        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graphs, n_nodes, n_hidden]

        # TODO: optimize into a single embedding table for bidirectional graphs
        node_feature += self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        graph_token_feature = self.graph_token.weight.expand(n_graphs, 1, -1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        num_heads: int,
        num_edge_embeddings: int,
        num_spatial_embeddings: int,
        num_edge_dist_embeddings: int,
        edge_type: str,
        multi_hop_max_dist: int,
        n_layers: int,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = FastLargeInputEmbeddingBag(num_edge_embeddings + 1, num_heads)
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dist_embeddings * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial_embeddings, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, attn_bias, spatial_pos, edge_input, attn_edge_type):
        graph_attn_bias = attn_bias.clone()

        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)  # head first
        graph_attn_bias[:, :, 1:, 1:] += spatial_pos_bias

        graph_token = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] += graph_token
        graph_attn_bias[:, :, 0, :] += graph_token

        # edge feature
        if self.edge_type == "multi_hop":
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = (spatial_pos - 1).clamp(min=1, max=self.multi_hop_max_dist)
                edge_input = edge_input[..., :self.multi_hop_max_dist, :]
            else:
                spatial_pos_ = (spatial_pos - 1).clamp(min=1)

            edge_input_shape = edge_input.shape
            edge_input = self.edge_encoder(edge_input.flatten(0, -2))
            edge_input = edge_input.view(*edge_input_shape[:-1], -1)

            max_dist = edge_input.shape[-2]

            edge_input = edge_input[..., None, :] @ \
                         self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, ...]
            edge_input = (edge_input.squeeze(-2).sum(-2) / (spatial_pos_.float().unsqueeze(-1)))
        else:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] += edge_input.permute(0, 3, 1, 2)  # head first

        return graph_attn_bias


class Structural3DFeatures(nn.Module):
    """ Transformer-M 3D node features and 3D edge bias """
    def __init__(
        self,
        num_kernels: int,
        num_embeddings: int,
        embedding_dim: int,
        num_heads: int,
        activation_fn: int):
        super().__init__()
        self.gbf = GBFEncoder(num_kernels, num_embeddings)
        self.edge_feats_proj = nn.Sequential(
            nn.Linear(num_kernels, num_kernels),
            get_activation_fn(activation_fn),
            nn.Linear(num_kernels, num_heads)
        )

        self.to_atom_feats = nn.Linear(num_kernels, embedding_dim)

    def forward(self, node_pos, atom_type_pair, padding_mask):
        delta_pos = node_pos.unsqueeze(1) - node_pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)
        delta_pos = delta_pos / dist.unsqueeze(-1).clamp_min(1e-12)

        dist_gbf = self.gbf(dist, atom_type_pair)
        edge_3d_bias = self.edge_feats_proj(dist_gbf)
        edge_3d_bias = edge_3d_bias.permute(0, 3, 1, 2).contiguous()  # heads first

        dist_gbf = dist_gbf.masked_fill(padding_mask[:, None, ..., None], 0.0)
        sum_dist_gbf = dist_gbf.sum(dim=-2)
        atom_3d_feats = self.to_atom_feats(sum_dist_gbf)

        return edge_3d_bias, atom_3d_feats, delta_pos


class NodePositionHead(nn.Module):
    """ Transformer-M node position head for denoising pretext """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool = True,
        equivariant: bool = False
    ):
        super().__init__()
        self.equivariant = equivariant
        self.partial_mha = PartialMultiheadAttention(embed_dim, num_heads, dropout, bias)

        if equivariant:
            self.force_proj = nn.Linear(embed_dim, 1, bias=False)
        else:
            self.force_proj = nn.ModuleList([
                nn.Linear(embed_dim, 1, bias=False),
                nn.Linear(embed_dim, 1, bias=False),
                nn.Linear(embed_dim, 1, bias=False),
            ])

    def forward(
        self,
        feats: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
    ):
        attn_probs, v = self.partial_mha(feats, attn_bias)
        v = v.view(*attn_bias.shape[:3], -1)
        attn_probs = attn_probs.view(*attn_bias.shape[:4])
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1)
        attn = torch.einsum('b h s t d, b h t c -> b s d h c', rot_attn_probs, v).flatten(-2, -1)

        if self.equivariant:
            out = self.force_proj(attn).squeeze(-1)
        else:
            out_x = self.force_proj[0](attn[..., 0, :])
            out_y = self.force_proj[1](attn[..., 1, :])
            out_z = self.force_proj[2](attn[..., 2, :])
            out = torch.cat([out_x, out_y, out_z], dim=-1)

        return out
