# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import pathlib
import sys

from graphormer.data_loading.dgl_datasets import DGL_DATASETS
from graphormer.data_loading.ogb_datasets import OGB_DATASETS
from graphormer.data_loading.pyg_datasets import PYG_DATASETS
from graphormer.runtime.criterions import CRITERION_MAP
from graphormer.model import GraphormerModel
from graphormer.runtime.utils import str2bool


def get_args():
    parser = argparse.ArgumentParser(description='Graphormer')

    paths = parser.add_argument_group('Paths')
    paths.add_argument('--data-dir', type=pathlib.Path, default=pathlib.Path('./data'),
                       help='Directory where the data is located or should be downloaded')
    paths.add_argument('--log-dir', type=pathlib.Path, default=pathlib.Path('/results'),
                       help='Directory where the results logs should be saved')
    paths.add_argument('--dllogger-name', type=str, default='dllogger_results.json',
                       help='Name for the resulting DLLogger JSON file')
    paths.add_argument('--prediction-name', type=str, default='predictions.pth',
                       help='Name for the resulting predictions dump')
    paths.add_argument('--save-ckpt-path', type=pathlib.Path, default=None,
                       help='File where the checkpoint should be saved')
    paths.add_argument('--load-ckpt-path', type=pathlib.Path, default=None,
                       help='File of the checkpoint to be loaded')

    optimizer = parser.add_argument_group('Optimizer')
    optimizer.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float, default=2e-4)
    optimizer.add_argument('--end-learning-rate', '--end-lr', dest='end_learning_rate', type=float, default=1e-9)
    optimizer.add_argument('--warmup-updates', '--warmup', dest='warmup_updates', type=int, default=0,
                           help='Number of updates for the learning rate to grow to learning_rate')
    optimizer.add_argument('--weight-decay', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--seed', type=int, default=None, help='Set a seed globally')

    parser.add_argument('--dataset-source', type=str, choices=['pyg', 'dgl', 'ogb'],
                        help='Source library where the dataset is loaded from')
    parser.add_argument('--dataset-name', type=str,
            help=f'''Name and options (<name>[:<options>]) for the dataset to use. Valid names:
                    OGB: {OGB_DATASETS}
                    PYG: {PYG_DATASETS}
                    DGL: {DGL_DATASETS}''',
            required=True)
    parser.add_argument('--num-workers', type=int, default=16, help='Number of dataloading workers')

    parser.add_argument('--amp', type=str2bool, nargs='?', const=True, default=False, help='Use Automatic Mixed Precision')
    parser.add_argument('--gradient-clip', type=float, default=None, help='Clipping of the gradient norms')
    parser.add_argument('--accumulate-grad-batches', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--ckpt-interval', type=int, default=-1, help='Save a checkpoint every N epochs')
    parser.add_argument('--eval-interval', dest='eval_interval', type=int, default=20,
                        help='Do an evaluation round every N epochs')
    parser.add_argument('--silent', type=str2bool, nargs='?', const=True, default=False,
                        help='Minimize stdout output')
    parser.add_argument('--benchmark', type=str2bool, nargs='?', const=True, default=False,
                        help='Benchmark mode')
    parser.add_argument('--wandb', type=str2bool, nargs='?', const=True, default=False,
                        help='Enable W&B logging')
    parser.add_argument('--cv-fold-path', type=pathlib.Path, default=None,
                        help='Path to custom folds (OGB only)')
    parser.add_argument('--cv-fold-idx', type=int, default=0, help='If using custom folds, fold index (OGB only)')

    parser.add_argument('--eval-metric', '--metric', dest='eval_metric', type=str, choices=['mae', 'auc'],
                        help='Metric to use during evaluation and test rounds')
    parser.add_argument('--criterion', type=str, nargs="+", choices=list(CRITERION_MAP.keys()),
                            help='Criterion to use during training')

    parser.add_argument('--max-nodes', type=int, default=128)
    parser.add_argument('--multi-hop-max-dist', type=int, default=5)
    parser.add_argument('--spatial-pos-max', type=int, default=1024)

    parser.add_argument('--finetune', type=str2bool, nargs='?', const=True, default=False,
                        help='Load a pretrained checkpoint to finetune on a new task')
    GraphormerModel.add_args(parser)

    args = parser.parse_args()

    return args, sys.argv
