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
import math
import pathlib
import random
import sys
import time
from typing import List
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from apex.contrib.clip_grad import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from graphormer.data_loading.dataset import GraphormerDataset
from graphormer.model.graphormer import GraphormerModel
from graphormer.runtime import gpu_affinity
from graphormer.runtime.arguments import get_args
from graphormer.runtime.criterions import GraphPredictionBinaryCrossEntropy
from graphormer.runtime.criterions import GraphPredictionL1Loss
from graphormer.runtime.criterions import GraphPredictionMulticlassCrossEntropy
from graphormer.runtime.criterions import CRITERION_MAP
from graphormer.runtime.criterions.metric import Metric
from graphormer.runtime.inference import evaluate
from graphormer.runtime.loggers import DLLogger
from graphormer.runtime.loggers import Logger
from graphormer.runtime.loggers import LoggerCollection
from graphormer.runtime.loggers import WandbLogger
from graphormer.runtime.utils import flatten_module_params
from graphormer.runtime.utils import get_local_rank
from graphormer.runtime.utils import increase_l2_fetch_granularity
from graphormer.runtime.utils import init_distributed
from graphormer.runtime.utils import seed_everything
from graphormer.runtime.utils import to_cuda


def save_state(model: nn.Module, optimizer: Optimizer, lr_scheduler: _LRScheduler, epoch: int,
               path: pathlib.Path):
    """ Saves model, optimizer and epoch states to path (only once per node) """
    if get_local_rank() == 0:
        state_dict = model.module.state_dict() if isinstance(
            model,
            DistributedDataParallel
        ) else model.state_dict()
        checkpoint = {
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch
        }

        torch.save(checkpoint, str(path))
        logging.info(f'Saved checkpoint to {str(path)}')


def load_state(model: nn.Module, optimizer: Optimizer, lr_scheduler: _LRScheduler,
               path: pathlib.Path, finetune: bool = False):
    """ Loads model, optimizer and epoch states from path """
    checkpoint = torch.load(str(path), map_location={'cuda:0': f'cuda:{get_local_rank()}'})
    if isinstance(model, DistributedDataParallel):
        model = model.module

    if finetune:
        del checkpoint['state_dict']['embed_out.weight']
        del checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    logging.info(f'Loaded checkpoint from {str(path)}')
    return checkpoint.get('epoch', 0)


def train_step(sample,
               model,
               criterions,
               grad_scaler,
               args):
    """
    Forward on the model and return the loss computed by criterion

    Args:
        sample (dict): mini-batch of data
        model (nn.Module): the model to be trained
        criterion  (Metric): the criterion to be optimized
        grad_scaler (torch.cuda.amp.GradScaler): grad scaler to scale and compute backward
    """

    targets = sample['y']
    with torch.cuda.amp.autocast(enabled=args.amp):
        out = model(sample)
    total_loss = 0.0
    for c in criterions:
        loss = c.update(**out, targets=targets, **sample)
        total_loss += loss
    grad_scaler.scale(total_loss).backward()
    return total_loss, {}


def train_epoch(model,
                criterions,
                optimizer,
                train_dataloader,
                epoch_idx,
                grad_scaler,
                lr_scheduler,
                logger,
                local_rank,
                args):
    model.train()
    module = model
    if isinstance(model, DistributedDataParallel):
        module = model.module

    timestamps = []
    progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit='batch',
                    desc=f'Epoch {epoch_idx}', disable=(args.silent or local_rank != 0))

    for i, batch in progress:
        if args.benchmark:
            torch.cuda.synchronize()
            timestamps.append(time.time())
            
        batch = to_cuda(batch)
        loss, logs = train_step(batch,
                                model,
                                criterions,
                                grad_scaler,
                                args)

        # gradient accumulation
        if (i + 1) % args.accumulate_grad_batches == 0 or (i + 1) == len(train_dataloader):
            module.sync_grad()
            norm = None
            if args.gradient_clip:
                grad_scaler.unscale_(optimizer)
                norm = clip_grad_norm_(module.flat_parameters(), args.gradient_clip)

            grad_scaler.step(optimizer)
            grad_scaler.update()
            module.zero_grad(set_to_none=True)
            lr_scheduler.step()

            logger.log_metrics(
                {'gradient_norm': None if norm is None else norm.item(),
                 'loss_scale': grad_scaler.get_scale(),
                 'learning_rate': lr_scheduler.get_last_lr()[0]},
                step=epoch_idx + i / len(train_dataloader)
            )

        progress.set_postfix({'loss': f'{loss.item():.4f}', **logs})

    return timestamps if args.benchmark else None


def train(model: nn.Module,
          criterions: List[Metric],
          optimizer: Optimizer,
          lr_scheduler: Optional[_LRScheduler],
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          logger: Logger,
          args):
    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    batch_times = []

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    epoch_start = load_state(model, optimizer, lr_scheduler,
                             args.load_ckpt_path, args.finetune) if args.load_ckpt_path and not args.benchmark else 0

    for epoch_idx in range(epoch_start, args.epochs):
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch_idx)

        for c in criterions:
            c.reset()
        times = train_epoch(model,
                            criterions,
                            optimizer,
                            train_dataloader,
                            epoch_idx,
                            grad_scaler,
                            lr_scheduler,
                            logger,
                            local_rank,
                            args)
        total_loss = 0.0
        all_logs = dict()
        for c in criterions:
            loss, logs = c.compute()
            total_loss += loss
            all_logs = {**logs, **all_logs}
        logs = all_logs
        logs['total_loss'] = total_loss
        logging.info(f'Train loss: {logs}')

        if math.isnan(total_loss) or math.isinf(total_loss):
            logging.error(f'Epoch loss was NaN or inf, exiting process {local_rank}')
            sys.exit(1)

        if not args.full_train and not args.benchmark and (
                (args.eval_interval > 0 and (
                        epoch_idx + 1) % args.eval_interval == 0) or epoch_idx + 1 == args.epochs):
            val_logs = evaluate(model, val_dataloader, local_rank, args)
            logging.info(f'Validation {args.eval_metric.upper()}: {val_logs[args.eval_metric]}')
            logs.update(val_logs)

        logger.log_metrics(logs, epoch_idx + 1)

        if not args.benchmark and args.save_ckpt_path is not None and args.ckpt_interval > 0 \
                and (epoch_idx + 1) % args.ckpt_interval == 0:
            save_state(model, optimizer, lr_scheduler, epoch_idx + 1, args.save_ckpt_path)


        if args.benchmark and epoch_idx >= 1:
            batch_times.extend(np.diff(times))

    if args.save_ckpt_path is not None and not args.benchmark:
        save_state(model, optimizer, lr_scheduler, args.epochs, args.save_ckpt_path)

    return batch_times if args.benchmark else None


def print_parameters_count(model):
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_params_trainable}')


def get_lr_scheduler(total_num_updates, optimizer, args):
    lr_range = args.learning_rate - args.end_learning_rate

    def polynomial_scheduler_with_warmup(num_updates):
        if args.warmup_updates > 0 and num_updates <= args.warmup_updates:
            return num_updates / args.warmup_updates
        else:
            remaining = 1 - (num_updates - args.warmup_updates) / (
                    total_num_updates - args.warmup_updates
            )
            lr = lr_range * remaining + args.end_learning_rate
            return lr / args.learning_rate

    return LambdaLR(
        optimizer,
        polynomial_scheduler_with_warmup,
    )


def get_model(a):
    a.num_atom_embeddings = 128 + 8 * 12  # matches ogb dataset, hopefully others are smaller
    a.num_edge_embeddings = 8 * 3  # same as above
    a.max_in_degree = a.max_nodes
    a.max_out_degree = a.max_nodes
    a.num_spatial_embeddings = a.spatial_pos_max + 1
    a.num_edge_dist_embeddings = a.multi_hop_max_dist + 1
    a.edge_type = 'multi_hop'
    a.remove_head = False
    return GraphormerModel(a)


def get_criterions(args):
    criterions = []
    for criterion in args.criterion:
        try:
            criterions.append(CRITERION_MAP[criterion](**vars(args)))
        except KeyError as e:
            raise e(f"Unknown criterion {args.criterion}")
    return criterions


if __name__ == '__main__':
    is_distributed = init_distributed()
    local_rank = get_local_rank()
    args, sargv = get_args()

    logging.getLogger().setLevel(
        logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO
    )

    logging.info('========== Graphormer =========')
    logging.info('|      Training procedure     |')
    logging.info('===============================')

    if args.seed is None:
        args.seed = random.randrange(2**30)
        if is_distributed:
            object_list = [args.seed]
            dist.broadcast_object_list(object_list, src=0)
            args.seed = object_list[0]

    logging.info(f'Using seed {args.seed}')
    seed_everything(args.seed)

    dataset = GraphormerDataset(
        dataset_source=args.dataset_source,
        dataset_spec=args.dataset_name,
        data_dir=args.data_dir,
        seed=args.seed,
        cv_fold_idx=args.cv_fold_idx,
        cv_fold_path=args.cv_fold_path,
        full_train=args.full_train,
    )
    train_dataloader = dataset.train_dataloader(
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_nodes=args.max_nodes, multi_hop_max_dist=args.multi_hop_max_dist,
        spatial_pos_max=args.spatial_pos_max,
    )
    val_dataloader = dataset.val_dataloader(
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_nodes=args.max_nodes, multi_hop_max_dist=args.multi_hop_max_dist,
        spatial_pos_max=args.spatial_pos_max,
    )

    model = get_model(args).cuda()
    flatten_module_params(model, args.amp)
    criterions = get_criterions(args)
    optimizer = AdamW(model.flat_parameters(),
                      lr=args.learning_rate,
                      betas=(0.9, 0.999),
                      weight_decay=args.weight_decay,
                      foreach=True)

    total_num_updates = int(
        math.ceil(len(train_dataloader) / args.accumulate_grad_batches * args.epochs)
    )
    lr_scheduler = get_lr_scheduler(total_num_updates, optimizer, args)

    loggers = [DLLogger(save_dir=args.log_dir, filename=args.dllogger_name)]
    if args.wandb:
        loggers.append(
            WandbLogger(
                name=f'{args.dataset_name}({args.dataset_source})',
                save_dir=args.log_dir,
                project='graphormer',
            )
        )
    logger = LoggerCollection(loggers)

    logging.info(f'Training for {args.epochs} epochs ({total_num_updates} steps)')
    logging.info(f'Training on dataset {args.dataset_name} from {args.dataset_source}')

    if is_distributed:
        gpu_affinity.set_affinity(gpu_id=get_local_rank(), nproc_per_node=torch.cuda.device_count(), scope='socket')

    torch.set_float32_matmul_precision('high')
    print_parameters_count(model)
    logger.log_hyperparams(vars(args))

    increase_l2_fetch_granularity()
    times = train(model,
                  criterions,
                  optimizer,
                  lr_scheduler,
                  train_dataloader,
                  val_dataloader,
                  logger,
                  args)

    logging.info('Training finished successfully')

    if args.benchmark:
        times = np.array(times) * 1000.0  # in ms
        total_batch_size = args.batch_size * (dist.get_world_size() if dist.is_initialized() else 1)
        logging.info(f'Training throughput: {(total_batch_size / times).mean()} mol/ms')
        logging.info(f'Training latency: {times.mean()} ms')
