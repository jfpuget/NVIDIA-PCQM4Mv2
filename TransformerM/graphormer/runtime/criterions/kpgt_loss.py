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

from pathlib import Path

import numpy
import torch
import torch.nn as nn
from torch import Tensor
from graphormer.runtime.criterions.metric import Metric

from graphormer.runtime.criterions.meta import FINGERPRINT_POS_WEIGHTS


class KPGTPretextLoss(Metric):
    def __init__(self, 
                 key="kpgt", 
                 kpgt_loss_weight_dc=0.1,
                 kpgt_loss_weight_fp=0.1,
                 **kwargs):
        super().__init__(f'kpgt_loss')
        self.key = key
        self.add_state('error', torch.tensor(0, dtype=torch.float32, device='cuda'))
        self.add_state('total', torch.tensor(0, dtype=torch.int32, device='cuda'))
        self.orig_vals = None

        self.amp = kwargs['amp']
        data_dir = Path(kwargs["data_dir"])
        fp_memmap = data_dir / 'pcqm-dpfp' / 'fingerprint.np'
        dc_memmap = data_dir / 'pcqm-dpfp' / 'descriptor.np'
        self.kpgt_dc_criterion = torch.nn.MSELoss()
        self.kpgt_fp_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=FINGERPRINT_POS_WEIGHTS.to('cuda'))

        self.kpgt_loss_weight_dc = kpgt_loss_weight_dc
        self.kpgt_loss_weight_fp = kpgt_loss_weight_fp
        self.fp_memmap = numpy.array(numpy.memmap(
            fp_memmap,
            dtype='float32',
            mode='r',
            shape=(3746620, 512)
        ))

        self.dc_memmap = numpy.array(numpy.memmap(
            dc_memmap,
            dtype='float32',
            mode='r',
            shape=(3746620, 201)
        ))

    def update(self, 
               kpgt: Tensor, 
               targets: Tensor, 
               idx: Tensor, **_):
        with torch.cuda.amp.autocast(enabled=self.amp):
            logits_dc, logits_fp = kpgt
            n = logits_fp.shape[0]

            targets_dc = torch.from_numpy(numpy.array(
                self.dc_memmap[idx.cpu().numpy(), 1:])).to(logits_dc.device)

            targets_fp = torch.from_numpy(numpy.array(
                self.fp_memmap[idx.cpu().numpy(), :])).to(logits_fp.device)

            loss_fp = self.kpgt_fp_criterion(logits_fp, targets_fp)
            nan_mask = torch.isnan(targets_dc)
            logits_dc = torch.where(nan_mask, 0, logits_dc)
            targets_dc = torch.where(nan_mask, 0, targets_dc)
            loss_dc = self.kpgt_dc_criterion(logits_dc, targets_dc)
            loss_kpgt = loss_fp * self.kpgt_loss_weight_fp + loss_dc * self.kpgt_loss_weight_dc

        error = loss_kpgt
        self.total += n
        self.error += error.detach()
        return loss_kpgt

    def _compute(self):
        return self.error / self.total, {}
