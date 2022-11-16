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

import logging

import torch
import torch.nn as nn

from graphormer.model.layers import GraphormerGraphEncoder
from graphormer.model.layers import init_graphormer_params
from graphormer.model.layers.nn_utils import get_activation_fn
from graphormer.model.layers.nn_utils import get_available_activation_fns
from graphormer.runtime.utils import str2bool

logger = logging.getLogger(__name__)


class GraphormerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.set_architecture_args(args)
        self.args = args

        self.encoder = GraphormerGraphEncoder(
            num_atom_embeddings=args.num_atom_embeddings,
            max_in_degree=args.max_in_degree,
            max_out_degree=args.max_out_degree,
            num_edge_embeddings=args.num_edge_embeddings,
            num_spatial_embeddings=args.num_spatial_embeddings,
            num_edge_dist_embeddings=args.num_edge_dist_embeddings,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            activation_fn=args.activation_fn,
            channel_prob_2d_only=args.channel_prob_2d_only,
            channel_prob_3d_only=args.channel_prob_3d_only,
            channel_prob_2d_3d=args.channel_prob_2d_3d,
            num_kernels=args.num_kernels,
            drop_path_prob=args.drop_path_prob,
            node_pos_head=args.node_pos_head,
            node_pos_attention_dropout=args.node_pos_attention_dropout,
            position_noise=args.position_noise,
            random_tp=args.random_tp,
            kpgt=(args.kpgt_loss_weight_dc or args.kpgt_loss_weight_fp)
        )

        self.head_transform = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.activation_fn = get_activation_fn(args.activation_fn)
        self.layer_norm = nn.LayerNorm(args.encoder_embed_dim)

        self.embed_out = None
        self.learned_bias_out = nn.Parameter(torch.zeros(1))  # TODO: why only one scalar?
        self.embed_out = nn.Linear(args.encoder_embed_dim, args.num_classes, bias=False)

        self.apply(init_graphormer_params)

    def forward(self, batched_data):
        out = self.encoder(batched_data)
        pred = out['pred']
        pred = pred.transpose(0, 1)  # batch dimension first
        pred = self.layer_norm(self.activation_fn(self.head_transform(pred)))

        if self.embed_out is not None:
            pred = self.embed_out(pred)
        if self.learned_bias_out is not None:
            pred = pred + self.learned_bias_out

        out['pred'] = pred
        return out

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--dropout',
            type=float,
            metavar='D',
            default=0.1,
            help='Dropout probability',
        )
        parser.add_argument(
            '--attention-dropout',
            type=float,
            metavar='D',
            default=0.1,
            help='Dropout probability for attention weights',
        )
        parser.add_argument(
            '--act-dropout',
            type=float,
            metavar='D',
            default=0.1,
            help='Dropout probability after activation in FFN',
        )
        # Arguments related to hidden states and self-attention
        parser.add_argument(
            '--encoder-ffn-embed-dim',
            type=int,
            metavar='N',
            default=80,
            help='Encoder embedding dimension for FFN',
        )
        parser.add_argument(
            '--encoder-layers',
            type=int,
            metavar='N',
            default=12,
            help='Number of encoder layers',
        )
        parser.add_argument(
            '--encoder-attention-heads',
            type=int,
            metavar='N',
            default=8,
            help='Number of encoder attention heads',
        )

        # Arguments related to input and output embeddings
        parser.add_argument(
            '--encoder-embed-dim',
            type=int,
            metavar='N',
            default=768,
            help='Encoder embedding dimension',
        )
        # Dirichlet task prob
        parser.add_argument(
            '--random-tp',
            type=str2bool, nargs='?', const=True, default=False,
            help='Task probs to follow dirichlet with task_probs',
        )
        # Regularization args
        parser.add_argument(
            '--kpgt-loss-weight-dc',
            type=float,
            default=0.0,
            help='KPGT loss weight dc',
        )
        parser.add_argument(
            '--kpgt-loss-weight-fp',
            type=float,
            default=0.0,
            help='KPGT loss weight FP',
        )

        tm = parser.add_argument_group('Transformer-M')
        tm.add_argument('--channel-prob-2d-only', type=float, default=0.2,
                        help='Probability for a training sample to use the 2D-only structural channel. '
                             'Set to 1.0 to match Graphormer behaviour.')
        tm.add_argument('--channel-prob-3d-only', type=float, default=0.6,
                        help='Probability for a training sample to use the 3D-only structural channel. '
                             'Set to 0.0 to match Graphormer behaviour.')
        tm.add_argument('--channel-prob-2d-3d', type=float, default=0.2,
                        help='Probability for a training sample to use the 2D and 3D structural channels. '
                             'Set to 0.0 to match Graphormer behaviour.')
        tm.add_argument('--position-noise', type=float, default=0.0,
                        help='Magnitude of the random node position perturbation')
        tm.add_argument('--num-kernels', type=int, default=128,
                        help='Number of RBF kernels to encode inter-atom distances')
        tm.add_argument('--drop-path-prob', type=float, default=0.0,
                        help='Maximum droppath probablity for residual connections')
        tm.add_argument('--node-pos-head', type=str2bool, nargs='?', const=True, default=False,
                        help='Add an additional head to predict node positions')
        tm.add_argument('--node-pos-attention-dropout', type=float, default=0.1,
                        help='Attention dropout in the node head')

        # misc params
        parser.add_argument(
            '--activation-fn',
            choices=get_available_activation_fns(),
            help='Activation function to use',
            default='gelu'
        )
        parser.add_argument('--num-classes', type=int, default=1)
        parser.add_argument('--architecture', '--arch', type=str, choices=['slim', 'base', 'medium-768', 'large'])

    @staticmethod
    def set_architecture_args(args):
        if not args.architecture:
            return

        if args.architecture == 'slim':
            args.encoder_layers = 12
            args.encoder_embed_dim = 80
            args.encoder_ffn_embed_dim = 80
            args.encoder_attention_heads = 8
        elif args.architecture == 'base':
            args.encoder_layers = 12
            args.encoder_embed_dim = 768
            args.encoder_ffn_embed_dim = 768
            args.encoder_attention_heads = 32
        elif args.architecture == 'medium-768':
            args.encoder_layers = 18
            args.encoder_embed_dim = 768
            args.encoder_ffn_embed_dim = 768
            args.encoder_attention_heads = 32
        elif args.architecture == 'large':
            args.encoder_layers = 24
            args.encoder_embed_dim = 1024
            args.encoder_ffn_embed_dim = 1024
            args.encoder_attention_heads = 32
        else:
            raise ValueError(f'Unknown architecture {args.architecture}')
