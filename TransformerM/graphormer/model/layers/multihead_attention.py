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

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


@torch.jit.script
def softmax_dropout(x, training: bool, dropout: float):
    y = F.softmax(x.float(), dim=-1)
    y = F.dropout(y, dropout, training)
    return y.type_as(x)


class PartialMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool = True
    ):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

    def forward(self, feats: Tensor, attn_bias: Tensor):
        seq_length = feats.shape[0]
        qkv_proj = self.qkv_proj(feats)

        # permute + contiguous on the merged tensor avoids multiple calls to contiguous later
        q, k, v = qkv_proj.view(seq_length, -1, self.head_dim, 3).permute(-1, 1, 0, 2).contiguous()

        attn_bias = attn_bias.view(-1, seq_length, seq_length)
        attn_weights = torch.baddbmm(attn_bias, q, k.transpose(-1, -2), alpha=self.scaling)
        attn_weights = attn_weights.view(-1, seq_length, seq_length)

        attn_probs = softmax_dropout(attn_weights, self.training, self.dropout)
        return attn_probs, v


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.partial_mha = PartialMultiheadAttention(embed_dim, num_heads, dropout, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, feats: Tensor, attn_bias: Tensor):
        attn_probs, v = self.partial_mha(feats, attn_bias)
        attn = torch.einsum('b s t, b t h -> s b h', attn_probs, v).reshape_as(feats)
        out = self.out_proj(attn)
        return out
