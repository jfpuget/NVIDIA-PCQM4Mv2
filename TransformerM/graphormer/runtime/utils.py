# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATESn. All rights reserved.
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
import ctypes
import datetime
import logging
import math
import os
import random
import types
from collections import defaultdict
from functools import wraps
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor


def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def to_cuda(x):
    """ Try to convert a Tensor, a collection of Tensors or a DGLGraph to CUDA """
    if isinstance(x, Tensor):
        return x.cuda(non_blocking=True)
    elif isinstance(x, tuple):
        return (to_cuda(v) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        # DGLGraph or other objects
        return x.to(device=torch.cuda.current_device())


def init_distributed() -> bool:
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1
    if distributed:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://',
                                timeout=datetime.timedelta(seconds=3600 * 2))
        if backend == 'nccl':
            torch.cuda.set_device(get_local_rank())
        else:
            logging.warning('Running on CPU only!')
        assert torch.distributed.is_initialized()
    return distributed


def get_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', 0))


def increase_l2_fetch_granularity():
    # maximum fetch granularity of L2: 128 bytes
    _libcudart = ctypes.CDLL('libcudart.so')
    # set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_zero_only(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


def itemize(obj):
    if isinstance(obj, dict):
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
    else:
        return [v.item() if isinstance(v, torch.Tensor) else v for v in obj]


def flatten_module_params(module, amp: bool = False):
    # does not handle
    # - shared params or modules
    # - params manually casted

    with torch.no_grad():
        autocast_dtype = torch.get_autocast_gpu_dtype()
        default_dtype = torch.get_default_dtype()

        autocast_modules = [
            nn.Linear, nn.Bilinear, nn.Conv2d, nn.Conv3d
        ] if amp else []
        param_infos = defaultdict(list)
        new_param_infos = defaultdict(list)
        flat_params = {}
        flat_grads = {}
        casted_flat_params = {}

        # collect parameters
        for m in module.modules():
            if isinstance(m, torch.jit.ScriptModule):
                # We can't modify parameters of a jitted module
                continue

            for n, p in m.named_parameters(recurse=False):
                if not p.requires_grad or p.dtype != default_dtype:
                    continue

                assert p.is_cuda
                if m.__class__ in autocast_modules:
                    param_infos[autocast_dtype].append((m, n, p))
                else:
                    param_infos[default_dtype].append((m, n, p))


        for dtype in param_infos.keys():
            alignment = 16 / getattr(
                torch, torch.storage._dtype_to_storage_type_map()[dtype]
            )().element_size()
            for m, n, p in param_infos[dtype]:
                new_attr = p.data.clone().type(dtype)
                new_attr.requires_grad_()
                delattr(m, n)
                # setattr(m, n, new_attr)
                new_param_infos[dtype].append((m, n, new_attr))

            flat_params[dtype] = nn.Parameter(
                torch.zeros(
                    sum([int(math.ceil(p.numel() / alignment) * alignment) for m, _, p in
                         param_infos[dtype]]),
                    device='cuda',
                    dtype=default_dtype,
                    requires_grad=True
                )
            )
            flat_grads[dtype] = torch.zeros_like(flat_params[dtype], dtype=dtype)
            casted_flat_param = flat_params[dtype].type(dtype)  # default_dtype will return same object
            casted_flat_params[dtype] = casted_flat_param

            offset = 0
            for m, n, p in new_param_infos[dtype]:
                copy = p.clone()
                param = nn.Parameter(p)

                param.set_(casted_flat_param._storage(), offset, p.size())
                param.copy_(copy)

                param.grad = torch.empty_like(p)
                param.grad.set_(flat_grads[dtype]._storage(), offset, param.grad.size())
                m.register_parameter(n, param)
                offset += int(math.ceil(p.numel() / alignment) * alignment)

            flat_params[dtype].copy_(casted_flat_param)

        def sync_grad(module, *_, **__):
            with torch.no_grad():
                for dtype, flat_grad in flat_grads.items():
                    # copy to full precision the populated (low precision) buffer and assign it as the grad of the master param
                    # so that the optimizer of gradient clipping can operate on it
                    flat_params[dtype].grad = flat_grad.float()

        def sync_param(module, *_, **__):
            # populate leaf params (casted)
            with torch.no_grad():
                for dtype, flat_param in flat_params.items():
                    if dtype != default_dtype:
                        casted_flat_params[dtype].copy_(flat_param)

        def sync_flat_params(module, *_, **__):
            with torch.no_grad():
                for dtype, flat_param in flat_params.items():
                    if dtype != default_dtype:
                        flat_param.copy_(casted_flat_params[dtype])

        def zero_grad(module, *_, **__):
            with torch.no_grad():
                for dtype, flat_grad in flat_grads.items():
                    # zero the actual (low precision) buffer autograd will populate
                    flat_grad.zero_()
                    flat_params[dtype].grad = None

        def flat_parameters(module, *_, **__):
            return flat_params.values()

        module.zero_grad = types.MethodType(zero_grad, module)
        module.sync_grad = types.MethodType(sync_grad, module)
        module.flat_parameters = types.MethodType(flat_parameters, module)

        module.register_forward_pre_hook(sync_param)
        module.register_load_state_dict_post_hook(sync_flat_params)

    return flat_params, flat_grads


