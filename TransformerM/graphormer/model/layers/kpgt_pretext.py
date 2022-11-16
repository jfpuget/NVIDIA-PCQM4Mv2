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
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from graphormer.model.layers.mlp import CustomMLPHead


@torch.jit.script
def softmax_dropout(x, training: bool, dropout: float):
    y = F.softmax(x.float(), dim=-1)
    y = F.dropout(y, dropout, training)
    return y.type_as(x)


class KPGTPretext(nn.Module):
    def __init__(
        self,
        model_dim: int,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.kpgt_dc_mlp = CustomMLPHead(
            input_dim=model_dim,
            output_dim=200,
            input_norm='none',
            input_activation='GELU',
            input_dropout=0.,
            norm='none',
            num_hidden_layers=2,
            hidden_dim=model_dim,
            activation='GELU'
        )
        self.kpgt_fp_mlp = CustomMLPHead(
            input_dim=model_dim,
            output_dim=512,
            input_norm='none',
            input_activation='GELU',
            input_dropout=0.,
            norm='none',
            num_hidden_layers=2,
            hidden_dim=model_dim,
            activation='GELU'
        )

    def forward(
        self,
        graph_reps,
        indices
    ) -> Tuple[Tensor, Tensor]:
        logits_dc = self.kpgt_dc_mlp(graph_reps)
        logits_fp = self.kpgt_fp_mlp(graph_reps)
        return logits_dc, logits_fp
