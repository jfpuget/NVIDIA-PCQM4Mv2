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

from functools import partial
from typing import Optional
from typing import Union

import pathlib
import torch.distributed as dist
from dgl.data import DGLDataset
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch_geometric.data import Data as PYGDataset

from .dgl_datasets import DGLDatasetLookupTable
from .dgl_datasets import GraphormerDGLDataset
from .ogb_datasets import OGBDatasetLookupTable
from .pyg_datasets import GraphormerPYGDataset
from .pyg_datasets import PYGDatasetLookupTable
from .utils import collator


class GraphormerDataset:
    def __init__(
        self,
        dataset: Optional[Union[PYGDataset, DGLDataset]] = None,
        dataset_spec: Optional[str] = None,
        dataset_source: Optional[str] = None,
        seed: int = 0,
        data_dir: str = '',
        cv_fold_idx: Optional[int] = None,
        cv_fold_path: Optional[pathlib.Path] = None,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
        full_train: bool = False,
    ):
        super().__init__()
        if dataset is not None:
            if dataset_source == "dgl":
                self.dataset = GraphormerDGLDataset(dataset, seed=seed, train_idx=train_idx,
                                                    valid_idx=valid_idx, test_idx=test_idx)
            elif dataset_source == "pyg":
                self.dataset = GraphormerPYGDataset(dataset, seed=seed, train_idx=train_idx,
                                                    valid_idx=valid_idx, test_idx=test_idx)
            else:
                raise ValueError("Customized dataset can only have source pyg or dgl")
        elif dataset_source == "dgl":
            self.dataset = DGLDatasetLookupTable.get_dataset(dataset_spec, seed=seed, data_dir=data_dir)
        elif dataset_source == "pyg":
            self.dataset = PYGDatasetLookupTable.get_dataset(dataset_spec, seed=seed, data_dir=data_dir)
        elif dataset_source == "ogb":
            self.dataset = OGBDatasetLookupTable.get_dataset(dataset_spec,
                                                             seed=seed,
                                                             data_dir=data_dir,
                                                             cv_fold_idx=cv_fold_idx,
                                                             cv_fold_path=cv_fold_path,
                                                             full_train=full_train)
        else:
            raise ValueError(f'Unknown dataset_source {dataset_source}')

        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.train_dataset = self.dataset.train_data
        self.val_dataset = self.dataset.valid_data
        self.test_dataset = self.dataset.test_data

    @staticmethod
    def _get_dataloader(dataset,
                        shuffle,
                        max_nodes=128,
                        multi_hop_max_dist=5,
                        spatial_pos_max=1024,
                        **kwargs):
        collate_fn = partial(collator,
                             max_nodes=max_nodes,
                             multi_hop_max_dist=multi_hop_max_dist,
                             spatial_pos_max=spatial_pos_max)

        kwargs = {
            'pin_memory': True,
            'persistent_workers': False, # True means sysmem OOM on PCQM4Mv1
            'collate_fn': collate_fn,
            **kwargs
        }
        # Classic or distributed dataloader depending on the context
        sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
        return DataLoader(dataset, shuffle=(shuffle and sampler is None), sampler=sampler, **kwargs)

    def train_dataloader(self, **kwargs):
        return self._get_dataloader(self.train_dataset, shuffle=True, **kwargs)

    def val_dataloader(self, **kwargs):
        return self._get_dataloader(self.val_dataset, shuffle=False, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._get_dataloader(self.test_dataset, shuffle=False, **kwargs)
