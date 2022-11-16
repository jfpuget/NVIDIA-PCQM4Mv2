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

import torch.distributed as dist
from torch_geometric.data import Dataset
from torch_geometric.datasets import *

from .pyg_dataset import GraphormerPYGDataset

PYG_DATASETS = [
        "qm7b",
        "qm9",
        "zinc",
        "moleculenet",
        ]


class PYGDatasetLookupTable:
    @staticmethod
    def get_inner_dataset(dataset_spec: str, **kwargs):
        if ":" in dataset_spec:
            [name, params] = dataset_spec.split(":")
            params = params.split(",")
        else:
            name = dataset_spec
            params = []

        data_dir = kwargs.get('data_dir', '')
        if not data_dir:
            data_dir = "dataset"
        args = dict(root=data_dir)
        if name == "qm7b":
            return QM7b(**args)
        elif name == "qm9":
            return QM9(**args)
        elif name == "zinc":
            train_set = ZINC(**args, split="train")
            valid_set = ZINC(**args, split="val")
            test_set = ZINC(**args, split="test")
            return train_set, valid_set, test_set
        elif name == "moleculenet":
            task = dict(map(lambda s: s.split("="), params))["name"]
            return MoleculeNet(**args, name=task)
        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")

    @staticmethod
    def get_dataset(dataset_spec: str, seed: int = 0,
                    data_dir: str = '', **kwargs) -> Dataset:
        if not dist.is_initialized() or dist.get_rank() == 0:
            PYGDatasetLookupTable.get_inner_dataset(dataset_spec,
                                                    data_dir=data_dir)
        if dist.is_initialized():
            dist.barrier()
        inner_dataset = PYGDatasetLookupTable.get_inner_dataset(dataset_spec,
                                                                data_dir=data_dir)
        if isinstance(inner_dataset, tuple):
            train_set, valid_set, test_set = inner_dataset
            return GraphormerPYGDataset(None, seed, None, None, None, train_set, valid_set, test_set, **kwargs)
        else:
            return GraphormerPYGDataset(inner_dataset, seed)
