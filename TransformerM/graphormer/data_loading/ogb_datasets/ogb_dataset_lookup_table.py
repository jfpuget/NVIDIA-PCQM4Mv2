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

import numpy as np
import torch
import torch.distributed as dist
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc import PygPCQM4MDataset
from ogb.lsc import PygPCQM4Mv2Dataset
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset

from ..pyg_datasets import GraphormerPYGDataset
from .pcqm4mv2_3d import Pyg3DPCQM4Mv2Dataset

OGB_DATASETS = [
        "ogbg-molhiv",
        "ogbg-molpcba",
        "pcqm4mv2-3d",
        "pcqm4mv2",
        "pcqm4m"
        ]


class OGBDatasetLookupTable:
    @staticmethod
    def get_inner_dataset(dataset_name: str, **kwargs):
        data_dir = kwargs.get('data_dir', '')
        if not data_dir:
            data_dir = "dataset"
        args = dict(root=data_dir)
        if dataset_name in ["ogbg-molhiv", "ogbg-molpcba"]:
            return PygGraphPropPredDataset(dataset_name, **args)
        elif dataset_name == "pcqm4mv2-3d":
            path = Path(args["root"]) / "pcqm4m-v2-3d"
            path.mkdir(parents=True, exist_ok=True)
            (path / "RELEASE_v1.txt").touch(exist_ok=True)
            return Pyg3DPCQM4Mv2Dataset(**args)
        elif dataset_name == "pcqm4mv2":
            path = Path(args["root"]) / "pcqm4m-v2"
            path.mkdir(parents=True, exist_ok=True)
            (path / "RELEASE_v1.txt").touch(exist_ok=True)
            return PygPCQM4Mv2Dataset(**args)
        elif dataset_name == "pcqm4m":
            path = Path(args["root"]) / "pcqm4m_kddcup2021"
            path.mkdir(parents=True, exist_ok=True)
            (path / "RELEASE_v1.txt").touch(exist_ok=True)
            return PygPCQM4MDataset(**args)
        else:
            raise ValueError(f"Unknown dataset name {dataset_name} for ogb source.")

    @staticmethod
    def get_dataset(dataset_name: str, seed: int = 0,
                    data_dir: str = '', **kwargs) -> Dataset:
        if not dist.is_initialized() or dist.get_rank() == 0:
            OGBDatasetLookupTable.get_inner_dataset(dataset_name,
                                                    data_dir=data_dir,
                                                    **kwargs)
        if dist.is_initialized():
            dist.barrier()
        inner_dataset = OGBDatasetLookupTable.get_inner_dataset(dataset_name,
                data_dir=data_dir, **kwargs)

        idx_split = inner_dataset.get_idx_split()

        if kwargs.get('cv_fold_path'):
            cv_fold = kwargs['cv_fold_idx']
            splits = torch.load(str(kwargs['cv_fold_path']))
            train_idx, valid_idx = splits[f'train_{cv_fold}'], splits[f'valid_{cv_fold}']
        else:
            train_idx = idx_split['train']
            valid_idx = idx_split["valid"]

        test_idx = idx_split["test-challenge" if "pcqm4mv2" in dataset_name else "test"]

        return GraphormerPYGDataset(inner_dataset, seed, train_idx, valid_idx, test_idx, **kwargs)
