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
from dgl.data import DGLDataset
from dgl.data import FakeNewsDataset
from dgl.data import GINDataset
from dgl.data import MiniGCDataset
from dgl.data import QM7bDataset
from dgl.data import QM9Dataset
from dgl.data import QM9EdgeDataset
from dgl.data import TUDataset

from .dgl_dataset import GraphormerDGLDataset

DGL_DATASETS = [
        "qm7b",
        "qm9edge*",
        "qm9*",
        "minigc*",
        "tu*",
        "gin*",
        "fakenews*",
        ]


class DGLDatasetLookupTable:
    @staticmethod
    def get_inner_dataset(dataset_name, **kwargs):
        raw_dir = kwargs.get('data_dir', '')
        if not raw_dir:
            raw_dir =  "/root/.dgl/"

        if ":" in dataset_name:
            params = dict(map(lambda s: s.split("="), dataset_name.split(":")[-1].split(",")))
        else:
            params = {}

        if dataset_name == "qm7b":
            return QM7bDataset(raw_dir=raw_dir)
        elif dataset_name.startswith("qm9edge"):
            label_keys = params["label_keys"].split("+")
            return QM9EdgeDataset(label_keys=label_keys, raw_dir=raw_dir)
        elif dataset_name.startswith("qm9"):
            label_keys = params["label_keys"].split("+")
            cutoff = float(params.get("cutoff", 5.0))
            return QM9Dataset(label_keys=label_keys, cutoff=cutoff, raw_dir=raw_dir)
        elif dataset_name.startswith("minigc"):
            num_graphs = int(params["num_graphs"])
            min_num_v = int(params["min_num_v"])
            max_num_v = int(params["max_num_v"])
            data_seed = int(params["seed"])
            return MiniGCDataset(num_graphs, min_num_v, max_num_v, seed=data_seed, raw_dir=raw_dir)
        elif dataset_name.startswith("tu"):
            return TUDataset(name=params["name"])
        elif dataset_name.startswith("gin"):
            nm = params["name"]
            self_loop = params["self_loop"] == "true"
            degree_as_nlabel = params["degree_as_nlabel"] == "true"
            return GINDataset(name=nm, self_loop=self_loop, degree_as_nlabel=degree_as_nlabel, raw_dir=raw_dir)
        elif dataset_name.startswith("fakenews"):
            return FakeNewsDataset(name=params["name"],
                                   feature_name=params["feature_name"],
                                   raw_dir=raw_dir)
        else:
            raise ValueError(f"Unknown dataset name {dataset_name} for dgl source.")

    @staticmethod
    def get_dataset(dataset_name: str, seed: int = 0,
                    data_dir: str = '', **kwargs) -> DGLDataset:
        if not dist.is_initialized() or dist.get_rank() == 0:
            DGLDatasetLookupTable.get_inner_dataset(dataset_name,
                                                    data_dir=data_dir,
                                                    **kwargs)
        if dist.is_initialized():
            dist.barrier()
        inner_dataset = DGLDatasetLookupTable.get_inner_dataset(dataset_name,
                                                                data_dir=data_dir,
                                                                **kwargs)

        return GraphormerDGLDataset(inner_dataset, seed)
