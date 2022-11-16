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

from copy import copy
from typing import List
from typing import Optional, Tuple

import numpy as np
import torch
from dgl import DGLGraph
from dgl.data import DGLDataset
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data as PYGGraph
from torch_geometric.data import Dataset

from .. import algos
from ..utils import convert_to_single_emb


class GraphormerDGLDataset(Dataset):
    def __init__(self,
        dataset: DGLDataset,
        seed: int = 0,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
    ):
        self.dataset = dataset
        self.seed = seed
        num_data = len(self.dataset)
        assert (
            (train_idx is None and valid_idx is None and test_idx is None) or
            (train_idx is not None and valid_idx is not None and test_idx is not None)
        )
        if train_idx is None:
            train_valid_idx, test_idx = train_test_split(
                np.arange(num_data), test_size=num_data // 10, random_state=seed
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=num_data // 5, random_state=seed
            )
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.__indices__ = None
        self.train_data = self.index_select(train_idx)
        self.valid_data = self.index_select(valid_idx)
        self.test_data = self.index_select(test_idx)

    def index_select(self, indices: List[int]):
        subset = copy(self)
        subset.__indices__ = indices
        subset.train_idx = None
        subset.valid_idx = None
        subset.test_idx = None
        subset.train_data = None
        subset.valid_data = None
        subset.test_data = None
        return subset

    def __extract_edge_and_node_features(
        self, graph_data: DGLGraph
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        def extract_tensor_from_node_or_edge_data(
            feature_dict: dict, num_nodes_or_edges
        ):
            float_feature_list = []
            int_feature_list = []

            def extract_tensor_from_dict(feature: torch.Tensor):
                if feature.dtype == torch.int32 or feature.dtype == torch.long:
                    int_feature_list.append(feature.unsqueeze(1))
                elif feature.dtype == torch.float32 or feature.dtype == torch.float64:
                    float_feature_list.append(feature.unsqueeze(1))

            for feature_or_dict in feature_dict:
                if isinstance(feature_or_dict, torch.Tensor):
                    extract_tensor_from_dict(feature_or_dict)
                elif isinstance(feature_or_dict, dict):
                    for feature in feature_or_dict:
                        extract_tensor_from_dict(feature)
            int_feature_tensor = (
                torch.from_numpy(np.zeros(shape=[num_nodes_or_edges, 1])).long()
                if len(int_feature_list) == 0
                else torch.cat(int_feature_list)
            )
            float_feature_tensor = (
                None if len(float_feature_list) == 0 else torch.cat(float_feature_list)
            )
            return int_feature_tensor, float_feature_tensor

        node_int_feature, node_float_feature = extract_tensor_from_node_or_edge_data(
            graph_data.ndata, graph_data.num_nodes()
        )
        edge_int_feature, edge_float_feature = extract_tensor_from_node_or_edge_data(
            graph_data.edata, graph_data.num_edges()
        )
        return (
            node_int_feature,
            node_float_feature,
            edge_int_feature,
            edge_float_feature,
        )

    def __preprocess_dgl_graph(
        self, graph_data: DGLGraph, y: torch.Tensor, idx: int
    ) -> PYGGraph:
        if not graph_data.is_homogeneous:
            raise ValueError(
                "Heterogeneous DGLGraph is found. Only homogeneous graph is supported."
            )
        num_nodes = graph_data.num_nodes()
        edge_index = graph_data.edges()
        node_int_feature, _, edge_int_feature, _ = self.__extract_edge_and_node_features(graph_data)

        attn_edge_type = torch.zeros([num_nodes, num_nodes, edge_int_feature.shape[1]],
                                     dtype=torch.long)
        attn_edge_type[edge_index[0].long(), edge_index[1].long()] \
            = convert_to_single_emb(edge_int_feature, 8)
        dense_adj = graph_data.adj().to_dense().int()

        shortest_path_result, path = algos.floyd_warshall(dense_adj.numpy())
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        spatial_pos = torch.from_numpy(shortest_path_result).long()
        attn_bias = torch.zeros([num_nodes + 1, num_nodes + 1], dtype=torch.float)  # with graph token

        pyg_graph = PYGGraph()
        pyg_graph.x = convert_to_single_emb(
            node_int_feature,
            [128] + [12] * (node_int_feature.shape[1] - 1)
        )
        pyg_graph.adj = dense_adj
        pyg_graph.attn_bias = attn_bias
        pyg_graph.attn_edge_type = attn_edge_type
        pyg_graph.spatial_pos = spatial_pos
        pyg_graph.in_degree = dense_adj.long().sum(dim=1).view(-1)
        pyg_graph.out_degree = pyg_graph.in_degree
        pyg_graph.edge_input = torch.from_numpy(edge_input).long()
        if y.dim() == 0:
            y = y.unsqueeze(-1)
        pyg_graph.y = y
        pyg_graph.idx = idx

        return pyg_graph

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError("index to a GraphormerDGLDataset can only be an integer.")

        if self.__indices__ is not None:
            idx = self.__indices__[idx]
        graph, y = self.dataset[idx]
        return self.__preprocess_dgl_graph(graph, y, idx)

    def __len__(self) -> int:
        return len(self.dataset) if self.__indices__ is None else len(self.__indices__)
