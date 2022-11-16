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

from abc import ABC
from abc import abstractmethod

import torch.distributed as dist
from torch import Tensor


class Metric(ABC):
    """ Metric class with synchronization capabilities similar to TorchMetrics """

    def __init__(self, name):
        self.states = {}
        self.name = name

    def add_state(self, name: str, default: Tensor):
        assert name not in self.states
        self.states[name] = default.clone()
        setattr(self, name, default)

    def synchronize(self):
        if dist.is_initialized():
            for state in self.states:
                dist.all_reduce(getattr(self, state), op=dist.ReduceOp.SUM, group=dist.group.WORLD)

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def reset(self):
        for name, default in self.states.items():
            setattr(self, name, default.clone())

    def compute(self):
        self.synchronize()
        value, logs = self._compute()
        logs[self.name] = value
        self.reset()
        return value.item(), {k: v.item() for k, v in logs.items()}

    @abstractmethod
    def _compute(self):
        pass

    @abstractmethod
    def update(self, sample):
        pass






