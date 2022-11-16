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

from typing import List, Union
import torch
import torch.nn


class CustomMLPHead(torch.nn.Module):
    """
    Custom Multi-layer Perceptron (fully-connected) head
    __Remark__: For corresponding use of convolution as mlp, please refer to the implementation
    of Gated Linear Unit as well.
    Parameters
    ----------
    input_dim: `int`, required
        input dimension
    hidden_dim: `Union[int, List[int]]`, required
        hidden layer dimension(s). pass `None` if `num_hidden_layers` is `0`.
    output_dim: `int`, required
        output dimension
    dropout: `float`, optional (default=0.0)
        dropout probability
    num_hidden_layers: `int`, optional (default=0)
        number of hidden layers (if 0, there will only be a layer to map input to output)
    norm: `str`, optional (default='LayerNorm')  # none, batchnorm, layernorm
        The torch normalization class to be applied to __hidden layers__.
    activation: `str`, optional (default='ReLU')  # torch.nn cclass or None
        The activation for __hidden layers__.
    input_norm: `str`, optional (default='none')
        The normalizer choice for application on __input layer__.
    output_norm: `str`, optional (default='none')
        The normalizer choice for application on __output layer__.
    output_activation: `str`, optional (default='none')
        The nonlinearity choice for application on __output layer__.
    output_dropout: `float`, optional (default=0.0)
        The dropout choice for application on __output layer__.
    input_activation: `str`, optional (default='none')
        The nonlinearity choice for application on __input layer__.
    input_dropout: `float`, optional (default=0.0)
        The dropout choice for application on __input layer__.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: Union[int, List[int]],
            output_dim: int,
            dropout: float = 0.0,
            num_hidden_layers: int = 0,
            norm: str = 'LayerNorm',  # none, batchnorm, layernorm
            activation: str = 'ReLU',  # torch.nn cclass or None
            input_norm: str = 'none',
            output_norm: str = 'none',
            output_activation: str = 'none',
            output_dropout: float = 0.0,
            input_activation: str = 'none',
            input_dropout: float = 0.0
    ):
        """constructor"""
        super().__init__()
        # - assertions
        if isinstance(hidden_dim, list):
            assert len(hidden_dim) == num_hidden_layers

        if num_hidden_layers == 0:
            assert hidden_dim == None

        self.pipeline = []

        # - gating the input
        if input_norm != 'none':
            self.pipeline.append(
                getattr(torch.nn, input_norm)(input_dim)
            )

        if input_activation != 'none':
            self.pipeline.append(
                getattr(torch.nn, input_activation)())

        if input_dropout > 0:
            self.pipeline.append(
                torch.nn.Dropout(input_dropout))

        last_dim = input_dim

        # - hidden layers (if any)
        if num_hidden_layers > 0:
            for i in range(num_hidden_layers):
                # - in dim
                if i == 0:
                    in_dim = input_dim
                else:
                    in_dim = hidden_dim if isinstance(hidden_dim, int) else hidden_dim[i - 1]

                out_dim = hidden_dim if isinstance(hidden_dim, int) else hidden_dim[i]
                last_dim = out_dim

                self.pipeline.append(
                    torch.nn.Linear(in_dim, hidden_dim if isinstance(hidden_dim, int) else hidden_dim[i]))
                if norm != 'none':
                    self.pipeline.append(getattr(torch.nn, norm)(out_dim))
                self.pipeline.append(getattr(torch.nn, activation)())
                if dropout > 0:
                    self.pipeline.append(torch.nn.Dropout(dropout))

        # - output gate
        self.pipeline.append(torch.nn.Linear(last_dim, output_dim))
        if output_norm != 'none':
            self.pipeline.append(
                getattr(torch.nn, output_norm)(output_dim)
            )

        if output_activation != 'none':
            self.pipeline.append(
                getattr(torch.nn, output_activation)())

        if output_dropout > 0:
            self.pipeline.append(
                torch.nn.Dropout(output_dropout))

        self.pipeline = torch.nn.Sequential(*self.pipeline)

    def forward(self, x):
        return self.pipeline(x)
