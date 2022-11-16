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

from typing import List

import torch.jit
import torch.nn as nn

from torch.distributions.dirichlet import Dirichlet
from torch.distributions.categorical import Categorical


def get_activation_fn(activation: str) -> nn.Module:
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "linear":
        return nn.Identity()
    elif activation == "swish":
        return nn.SiLU()
    else:
        raise RuntimeError(f"{activation} activation not supported")


def get_available_activation_fns() -> List[str]:
    return [
        "relu",
        "gelu",
        "tanh",
        "linear",
    ]


@torch.jit.script
def gaussian_basis_function(x, means, stds, epsilon: float = 1e-2):
    stds = stds.abs().clamp(min=epsilon)
    return (- 0.5 * ((x - means) / stds) ** 2).exp() / (stds * (2 * torch.pi) ** 0.5)


class RandomTP():
    """Sample channel masks with Dirichlet distribution concentration 
       [channel_prob_2d_only, channel_prob_2d_3d, channel_prob_3d_only]
       
       Args:
            channel_prob_2d_only (float): probability concetration for 2d channel
            channel_prob_2d_3d (float): probability concetration for both 2d/3d channel
            channel_prob_3d_only (float): probability concetration for 3d channel
    """
    def __init__(self, 
                 channel_prob_2d_only: float, 
                 channel_prob_2d_3d: float, 
                 channel_prob_3d_only: float):

        concentration = torch.Tensor([channel_prob_2d_only, 
                                      channel_prob_2d_3d, 
                                      channel_prob_3d_only])

        self.dirichlet = Dirichlet(concentration)
    
    def sample(size):
        dist = self.dirichlet.sample()
        channel_dist = Categorical(probs=dist)
        return channel_dist.sample(size)