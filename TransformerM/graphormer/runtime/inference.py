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

import logging
import random
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphormer.runtime.arguments import get_args
from graphormer.runtime.utils import flatten_module_params
from graphormer.runtime.utils import get_local_rank
from graphormer.runtime.utils import increase_l2_fetch_granularity
from graphormer.runtime.utils import seed_everything
from graphormer.runtime.utils import to_cuda


@torch.inference_mode()
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             local_rank: int,
             args):
    model.eval()
    y_pred, y_true = [], []
    progress = tqdm(dataloader, total=len(dataloader), unit='batch',
                    desc='Evaluation', disable=(args.silent or local_rank != 0))
    with torch.cuda.amp.autocast(enabled=args.amp):
        for batch in progress:
            batch = to_cuda(batch)
            preds = model(batch)['pred'][:, 0, :]
            y_pred.append(preds.detach().flatten().cpu())
            y_true.append(batch['y'].flatten().cpu()[:preds.shape[0]])

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    torch.save(y_pred, str(args.log_dir / args.prediction_name))

    # TODO: compute metrics on GPU?
    if args.eval_metric == 'auc':
        metric = torch.as_tensor(roc_auc_score(y_true, y_pred))
    elif args.eval_metric == 'mae':
        metric = (y_true - y_pred).abs().mean()
    else:
        raise ValueError(f'Unsupported metric {args.eval_metric}')

    if dist.is_initialized():
        metric = metric.cuda()
        dist.all_reduce(metric, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        metric = metric.cpu()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        metric /= world_size

    return {args.eval_metric: metric.item()}


if __name__ == '__main__':
    from graphormer.data_loading.dataset import GraphormerDataset
    from graphormer.runtime.loggers import DLLogger, LoggerCollection, WandbLogger
    from graphormer.runtime.training import get_model

    local_rank = get_local_rank()
    args, sargv = get_args()

    logging.getLogger().setLevel(
        logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO
    )

    logging.info('========== Graphormer =========')
    logging.info('|  Inference on the test set  |')
    logging.info('===============================')

    if not args.benchmark and args.load_ckpt_path is None:
        logging.error('No load_ckpt_path provided, you need to provide a saved model to evaluate')
        sys.exit(1)

    loggers = [DLLogger(save_dir=args.log_dir, filename=args.dllogger_name)]
    if args.wandb:
        loggers.append(
            WandbLogger(
                name=f'{args.dataset_name}({args.dataset_source})',
                save_dir=args.log_dir,
                project='graphormer'
            )
        )
    logger = LoggerCollection(loggers)

    if args.seed is None:
        args.seed = random.randrange(2**30)

    logging.info(f'Using seed {args.seed}')
    seed_everything(args.seed)

    model = get_model(args)
    model.to(device=torch.cuda.current_device())

    if args.load_ckpt_path is not None:
        checkpoint = torch.load(str(args.load_ckpt_path),
                                map_location={'cuda:0': f'cuda:{local_rank}'})
        model.load_state_dict(checkpoint['state_dict'])

    flatten_module_params(model, args.amp)
    dataset = GraphormerDataset(
        dataset_source=args.dataset_source,
        dataset_spec=args.dataset_name,
        data_dir=args.data_dir,
        seed=args.seed,
        cv_fold_idx=args.cv_fold_idx,
        cv_fold_path=args.cv_fold_path,
    )
    dataloader = dataset.test_dataloader(
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_nodes=args.max_nodes, multi_hop_max_dist=args.multi_hop_max_dist,
        spatial_pos_max=args.spatial_pos_max,
    )

    logger.log_hyperparams(vars(args))

    torch.set_float32_matmul_precision('high')
    increase_l2_fetch_granularity()
    logs = evaluate(model,
                    dataloader,
                    local_rank,
                    args)

    logger.log_metrics(logs)
    for k, v in logs.items():
        logging.info(f'Test {k.upper()}: {v}')
