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


class AtomDenoiseLoss(Metric):
    def __init__(self, key="node_pos_head", **kwargs):
        super().__init__(f'denoise_loss')
        self.key = key
        self.add_state('error', torch.tensor(0, dtype=torch.float32, device='cuda'))
        self.add_state('total', torch.tensor(0, dtype=torch.int32, device='cuda'))
        self.orig_vals = None

    def update(self, 
               node_pos_head: Tensor,
               pos_noise: Tensor,
               channels_3d_mask: Tensor, 
               targets: Tensor, 
               node_pos: Tensor, **_):

        if node_pos_head is None:
            return 0.0
        n = node_pos_head.shape[0]
        mask = (node_pos == 0.0).all(dim=-1, keepdim=True)[channels_3d_mask]
        pos_noise = pos_noise[channels_3d_mask]
        masked_noise = pos_noise.masked_fill(mask, 0.0)
        preds = node_pos_head.to(torch.float32)
        preds_mask = (preds == 0.0).all(dim=-1).all(dim=-1)[:, None, None] + mask
        noise_masked = masked_noise.masked_fill(preds_mask, 0.0).to(torch.float32)
        loss = 1.0 - nn.functional.cosine_similarity(preds, noise_masked, dim=-1)
        loss.masked_fill_(preds_mask.squeeze(-1), 0.0)
        loss = loss.sum(dim=-1)
        n_nnm = (~preds_mask).squeeze(-1).sum(dim=-1).to(loss)
        n_nnm = n_nnm.masked_fill(n_nnm == 0.0, 1.0)
        error = (loss / n_nnm).sum()
        if torch.isnan(error):
            error = torch.tensor(0.0).to(preds.device)
        else:
            self.total += n
            self.error += error.detach()
            error = error / n
        return error

    def _compute(self):
        self.total = max(self.total, 1)
        return self.error / self.total, {}