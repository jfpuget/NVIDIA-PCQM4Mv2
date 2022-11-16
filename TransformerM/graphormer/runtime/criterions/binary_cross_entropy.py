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
import torch.nn as nn
from torch import Tensor

from graphormer.runtime.criterions.metric import Metric


class GraphPredictionBinaryCrossEntropy(Metric):
    def __init__(self, **kwargs):
        super().__init__('binary_cross_entropy')
        self.add_state('error', torch.tensor(0, dtype=torch.float32, device='cuda'))
        self.add_state('total', torch.tensor(0, dtype=torch.int32, device='cuda'))
        self.add_state('correct', torch.tensor(0, dtype=torch.int32, device='cuda'))

    def update(self, pred: Tensor, targets: Tensor, **_):
        preds = pred[:, 0, :]
        binary_preds = (torch.sigmoid(preds) < 0.5).int()
        preds_flatten = preds.flatten()
        targets_flatten = targets.flatten()
        mask = ~torch.isnan(targets_flatten)
        error = nn.functional.binary_cross_entropy_with_logits(
            preds_flatten[mask].float(), targets_flatten[mask].float(), reduction='sum'
        )
        total = mask.int().sum()
        self.total += total
        self.error += error.detach()
        self.correct += (binary_preds.flatten()[mask] == targets_flatten[mask]).detach().sum()

        return error / total

    def _compute(self):
        return self.error / self.total, {'accuracy': self.correct / self.total}
