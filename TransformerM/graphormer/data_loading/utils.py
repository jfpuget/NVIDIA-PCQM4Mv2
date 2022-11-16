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
from typing import Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from . import algos


def pad_attn_bias(input_x, padlen):
    # - assumes all x's hve same dim
    new_x = torch.zeros([len(input_x), padlen, padlen],
                        dtype=input_x[0].dtype).fill_(float("-inf"))
    for i, x in enumerate(input_x):
        xlen = x.shape[0]
        if xlen < padlen:
            new_x[i, :xlen, :xlen] = x
            new_x[i, xlen:, :xlen] = 0
        else:
            new_x[i] = x
    return new_x


def pad_1d_plus1(input_x, padlen):
    new_x = torch.zeros([len(input_x),
                         padlen],
                        dtype=input_x[0].dtype)
    for i, x in enumerate(input_x):
        x = x + 1  # pad id = 0
        xlen = x.size(0)
        if xlen < padlen:
            new_x[i, :xlen] = x
        else:
            new_x[i] = x[:padlen]
    return new_x


def pad_2d(input_x, padlen):
    new_x = torch.zeros([len(input_x),
                         padlen,
                         padlen,
                         input_x[0].shape[-1]],
                         dtype=input_x[0].dtype)

    for i, x in enumerate(input_x):
        xlen = x.size(0)
        new_x[i, :xlen, :xlen, :] = x

    return new_x


def pad_spatial_pos(input_x, padlen):
    new_x = torch.zeros([len(input_x), padlen, padlen],
            dtype=input_x[0].dtype)
    for i, x in enumerate(input_x):
        x = x + 1
        xlen = x.size(0)
        new_x[i, :xlen, :xlen] = x
    return new_x


def pad_3d(input_x, padlen1, padlen2, padlen3):
    new_x = torch.zeros([len(input_x),
                         padlen1,
                         padlen2,
                         padlen3,
                         input_x[0].shape[-1]],
                         dtype=input_x[0].dtype)
    for i, x in enumerate(input_x):
        x = x + 1
        xlen1, xlen2, xlen3, xlen4 = x.size()
        new_x[i, :xlen1, :xlen2, :xlen3, :] = x.unsqueeze(0)
    return new_x


def collator(items,
             max_nodes=512,
             multi_hop_max_dist=20,
             spatial_pos_max=20):

    items = [item for item in items if item is not None and item.x.size(0) <= max_nodes]
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.node_type_pair if hasattr(item, 'node_type_pair') else None,
            item.node_pos if hasattr(item, 'node_pos') else None,
            item.y,
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        node_type_pair,
        node_pos,
        ys,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = pad_sequence(xs, batch_first=True)
    x = torch.where(x > 0, x + 1, x)

    out = dict()
    if node_pos[0] is not None:
        out['node_pos'] = pad_sequence(node_pos, batch_first=True)

    max_node_num = x.shape[1]
    edge_input = pad_3d(edge_inputs, max_node_num, max_node_num, max_dist)
    attn_bias = pad_attn_bias(attn_biases, max_node_num + 1)
    attn_edge_type = pad_2d(attn_edge_types, max_node_num)
    if node_type_pair[0] is not None:
        out['node_type_pair'] = pad_2d(node_type_pair, max_node_num)

    spatial_pos = pad_spatial_pos(spatial_poses, max_node_num)
    in_degree = pad_1d_plus1(in_degrees, max_node_num)

    out.update(dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,
    ))
    return out


@torch.jit.script
def convert_to_single_emb(x, offset: Union[int, List [int]]):
    feature_num = 1 if len(x.shape) < 1 else x.shape[1]
    if isinstance(offset, int):
        feature_offset = torch.arange(0, feature_num * offset, offset, dtype=torch.long) + 1
    else:
        feature_offset = torch.as_tensor(offset, dtype=torch.long).cumsum(dim=0) + 1
    x = x + feature_offset
    return x.long()


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x, [0, 128] + [12] * (x.shape[1] - 2 if len(x.shape) > 1 else 0))

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
            convert_to_single_emb(edge_attr, 8) + 1
    )

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy(shortest_path_result).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    node_type_pair = torch.cat(torch.broadcast_tensors(
        x[None, :, 0, None],
        128 + x[:, None, 0, None]
    ), dim=-1)
    node_type_pair[node_type_pair == 128] = 0

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()
    item.node_type_pair = node_type_pair

    return item
