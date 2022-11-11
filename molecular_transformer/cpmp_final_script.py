#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import argparse
import os
import sys
from pathlib import Path

def init_config(cuda_devices, fname, fold, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices

    DP = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) 

    CONFIG = {"fname" : fname,
              "seed": 2021,
              "epochs": 30,
              "train_batch_size": 64,
              "valid_batch_size": 64,
              "n_accumulate" : 1,
              "max_nodes" : 51,
              "max_edges" : 59,
              "max_rings" : 15,
              "num_node_feat" : 9,
              "num_edge_feat" : 3,
              "max_ring_length" : 20,
              "target_mean" : 5.68,
              "learning_rate": 1e-4,
              'head_learning_rate': 1e-4,
              "scheduler": 'SineAnnealingLR',
              "min_lr": 1e-8, 
              'weight_decay' : 0.05,
              'head_decay' : 0.05,
              'opt_beta1' : 0.9,
              'opt_beta2' : 0.999,
              'opt_eps' : 1e-8, 
              'verbose' : True,
              'workers' : 4,
              'hidden_size' : 768,
              'num_layer' : 12,
              'n_head' : 32,
              'fast_att' : 10,
              'max_degree' : 6,
              'grad_norm' : 0.0,
              'fold' : fold,
              }

    if CONFIG['fold'] is not None:
        fname = fname + ('_%d' % CONFIG['fold'])
        print('new fname', fname)
        CONFIG['fname'] = fname        

    checkpoint_path = Path(output_dir) / fname

    if not checkpoint_path.exists():
        checkpoint_path.mkdir()
    else:
        print('checkpoint path exists:', fname)

    if DP > 0:
        CONFIG['workers'] = DP * CONFIG['workers']
        CONFIG['train_batch_size'] = DP * CONFIG['train_batch_size']
        CONFIG['valid_batch_size'] = DP * CONFIG['valid_batch_size']
        #CONFIG['learning_rate'] = DP * CONFIG['learning_rate']
        #CONFIG['head_learning_rate'] = DP * CONFIG['head_learning_rate']

    DP = DP > 1
    print('DP', DP)
    CONFIG['dp'] = DP

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    return CONFIG

import gc
import copy
import time
import random
import string

# For data manipulation
import numpy as np
import pandas as pd
from scipy.special import softmax, expit, logit

from rdkit import Chem
from tqdm import tqdm

from ogb.lsc import PCQM4Mv2Dataset
from ogb.utils import smiles2graph
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 
from ogb.utils.features import atom_to_feature_vector
from ogb.utils.features import bond_to_feature_vector
from ogb.utils.url import download_url
from ogb.utils.url import extract_zip

import os.path as osp
import shutil
import tarfile

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint


def seed_torch(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def smiles2graphring(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)
        ssr = Chem.GetSymmSSSR(mol)
        ssr = [list(sr) for sr in ssr]

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)
        ssr = []

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    graph['ssr'] = ssr

    return graph 

def get_data(root):
    dataset = PCQM4Mv2Dataset(root=root, smiles2graph=smiles2graphring)
    return dataset


def get_split(dataset, split_path, CONFIG):
    
    split_dict = dataset.get_idx_split()
    train_idx = split_dict['train'] # pytorch tensor storing indices of training molecules
    valid_idx = split_dict['valid'] # pytorch tensor storing indices of validation molecules
    testdev_idx = split_dict['test-dev'] # pytorch tensor storing indices of test-dev molecules
    testchallenge_idx = split_dict['test-challenge'] # pytorch tensor storing indices of test-challenge molecules
    

    new_split_dict = torch.load(split_path)
                                       
    if CONFIG['fold'] is not None:
        train_idx = new_split_dict['train_%d' % CONFIG['fold']]
        valid_idx = new_split_dict['valid_%d' % CONFIG['fold']]
        print('using split for fold', CONFIG['fold'])
    
    return train_idx, valid_idx           

class Dataset():
    def __init__(self, data, index, aug, CONFIG):
        super(Dataset, self).__init__()
        self.max_nodes = CONFIG['max_nodes']
        self.num_node_feat = CONFIG['num_node_feat']
        self.max_edges = CONFIG['max_edges']
        self.num_edge_feat = CONFIG['num_edge_feat']
        self.max_rings = CONFIG['max_rings']
        self.fast_att = CONFIG['fast_att']
        self.aug = False #aug
                
        res = [data[idx] for idx in tqdm(index)]
        self.graphs = np.array([r[0] for r in res])
        self.targets = np.array([r[1] for r in res])

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        elt = self.graphs[idx]
        target = self.targets[idx]
        num_nodes = elt['num_nodes']
        elt_node_feat = elt['node_feat']
        elt_edge_feat = elt['edge_feat']
        edge_feat_tight = elt_edge_feat[::2]
        num_edges = edge_feat_tight.shape[0]
        edge_index = elt['edge_index']
        ssr = elt['ssr']
        num_rings = len(ssr)
        max_num_nodes = num_nodes
        max_num_edges = num_edges
        max_num_rings = num_rings

        node_feat = elt_node_feat           

        degrees = np.zeros(num_nodes)
        e_idx = edge_index[0]
        for i in range(len(e_idx)):
            degrees[e_idx[i]] += 1

        edge_feat = edge_feat_tight
        edge_node_mask = np.ones(num_edges)
                    
        max_num_nodes += max_num_edges
        max_num_nodes += max_num_rings        

        edge_mask_tight = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        edge_mask_tight[edge_index[0], edge_index[1]] = 1

        edge_mask = np.zeros((max_num_nodes + 1, max_num_nodes + 1), dtype=np.float32)
        edge_mask[1 : 1 + num_nodes, 1 : 1 + num_nodes] = edge_mask_tight
        edge_mask[0, 1:1 + num_nodes] = 1
        edge_mask[1:1 + num_nodes, 0] = 1
        edge_mask[0, 0] = 0

        x = np.arange(num_edges) + num_nodes + 1
        edge_index_tight = edge_index[:, ::2]
        edge_mask[edge_index_tight[0], x] = 1
        edge_mask[x, edge_index_tight[0]] = 1
        edge_mask[edge_index_tight[1], x] = 1
        edge_mask[x, edge_index_tight[1]] = 1
        edge_mask[0, 1 + num_nodes : 1 + num_nodes + num_edges] = 1
        edge_mask[1 + num_nodes : 1 + num_nodes + num_edges, 0] = 1
        num_rings = len(ssr)
        j = num_edges + num_nodes + 1
        for sr in ssr:
            for i in sr:
                edge_mask[i+1, j] = 1
                edge_mask[j, i+1] = 1
            edge_mask[0, j] = 1
            edge_mask[j, 0] = 1                                   
            j += 1
        ring_lengths = np.zeros(self.max_rings, dtype=np.int64)
        for i,sr in enumerate(ssr):
            ring_lengths[i] = len(sr)
            
                            
        batch_dict = {}
        batch_dict['num_nodes'] = torch.tensor([num_nodes], dtype=torch.long)
        batch_dict['num_edges'] = torch.tensor([num_edges], dtype=torch.long)
        batch_dict['edge_mask'] = torch.tensor(edge_mask, dtype=torch.float)
        batch_dict['node_feat'] = torch.tensor(node_feat.T, dtype=torch.long) # transpose for atom encoder
        batch_dict['edge_feat'] = torch.tensor(edge_feat.T, dtype=torch.long) # transpose for bond encoder
        batch_dict['edge_node_mask'] = torch.tensor(edge_node_mask, dtype=torch.float)
        batch_dict['target'] = torch.tensor([target], dtype=torch.float)
        batch_dict['num_rings'] = torch.tensor([num_rings], dtype=torch.long)
        batch_dict['ring_lengths'] = torch.tensor(ring_lengths, dtype=torch.long)
        batch_dict['degrees'] = torch.tensor(degrees, dtype=torch.long)
        return batch_dict

def collate_truncate(batch):
    # print('batch', len(batch))
    batch_dict = {}
    elt_0 = batch[0]
    for key in elt_0:
        n = len(elt_0[key].shape)
        if n == 1:
            d1 = max(b[key].shape[0] for b in batch)
            #print(key, d1)
            batch_dict[key] = torch.stack([F.pad(b[key], (0, d1 - b[key].shape[0])) for b in batch])
        elif n == 2:
            d1 = max(b[key].shape[0] for b in batch)
            d2 = max(b[key].shape[1] for b in batch)
            #print(key, d1, d2)
            batch_dict[key] = torch.stack([F.pad(b[key], (0, d2 - b[key].shape[1], 0, d1 - b[key].shape[0])) for b in batch])
    return batch_dict   

def get_data_loader(data, index, shuffle, CONFIG,):
    if shuffle:
        batch_size = CONFIG['train_batch_size']
    else:
        batch_size = CONFIG['valid_batch_size']
    dataset = Dataset(data, index, aug=False, CONFIG=CONFIG,)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=CONFIG['workers'],
        shuffle=shuffle,
        pin_memory=False,
        #worker_init_fn=worker_init_fn,
        collate_fn=collate_truncate,
        drop_last = shuffle,
    )
    return data_loader


def glorot_uniform(parameter):
    nn.init.xavier_uniform_(parameter.data, gain=1.0)

class ConvBloc1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBloc1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(in_channels)
        self.conv1 = nn.Linear(in_channels, out_channels, bias=False)
        glorot_uniform(self.conv1.weight)
        self.norm1 = nn.LayerNorm(out_channels)
        self.activate1 = nn.GELU()            
        self.conv2 = nn.Linear(out_channels, out_channels, bias=False)
        glorot_uniform(self.conv2.weight)
        self.norm2 = nn.LayerNorm(out_channels)
        self.activate2 = nn.GELU()            
        
    def forward(self, nodes):
        nodes = self.norm(nodes)
        x = self.conv1(nodes)
        x = self.norm1(x)
        x = self.activate1(x)
        nodes = nodes + x
        x = self.conv2(nodes)
        x = self.norm2(x)
        x = self.activate2(x)
        nodes = nodes + x
        return nodes

class FastMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_head, fast_att, ):
        super(FastMultiHeadAttention, self).__init__()
        self.fast_att = fast_att
        self.conv_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.conv_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.conv_V = nn.Linear(embedding_dim, embedding_dim, bias=False)
        glorot_uniform(self.conv_Q.weight)
        glorot_uniform(self.conv_K.weight)
        glorot_uniform(self.conv_V.weight)

        if n_head > 1:
            self.lin = nn.Linear(embedding_dim, embedding_dim, bias=False)
            glorot_uniform(self.lin.weight)
            n_factor = embedding_dim // n_head
            self.n_factor = n_factor
            self.n_head = n_head            
        else:
            self.att_head = Attention(embedding_dim, embedding_dim) 
        self.ln = nn.LayerNorm([embedding_dim])
        
    def forward(self, x_inner, x_outer, mask, dist):
        if self.n_head > 1:
            b, n, d = x_inner.shape
            h = self.n_head
            f = self.n_factor
            Q = self.conv_Q(x_inner) / np.sqrt(self.n_factor)   # b x n x d
            Q = Q.view(b, n, h, f)                              # b x n x h x f (h*f = d)
            Q = Q.transpose(1, 2).contiguous()                  # b x h x n x f
            Q = Q.view(b * h, n, f)                             # bh x n x f
            K = self.conv_K(x_outer)                            # b x n x d
            K = K.view(b, n, h, f)                              # b x n x h x f
            K = K.movedim(1, -1).contiguous()                   # b x h x f x n
            K = K.view(b * h, f, n)                             # bh x f x n
            V = self.conv_V(x_outer)                            # b x n x d
            V = V.view(b, n, h, f)                              # b x n x h x f
            V = V.transpose(1, 2).contiguous()                  # b x h x n x f
            V = V.view(b * h, n, f)                             # bh x n x f
            att = torch.bmm(Q, K)                               # bh x n x n
            mask = mask.unsqueeze(1)                            # b x 1 x n x n
            mask = mask.repeat(1, h, 1, 1)                      # b x h x n x n
            mask = mask.view(b * h, n, n)                       # bh x n x n
            if self.fast_att:               
                att = att - mask
                att = F.softmax(att, -1)
            else:
                att = mask * att - 10 * (1 - mask)
                att = F.softmax(att, -1) * mask
            att = torch.bmm(att, V)    # bh x n x f
            att = att.view(b, h, n, f) # b x h x n x f
            att = att.transpose(1, 2).contiguous()  # b x n x h x f
            att = att.view(b, n, h*f)   # b x n x d
            att = self.lin(att)         
        else:
            att = self.att_head(x_inner, x_outer, mask, dist)
        x = x_inner + att
        x = self.ln(x)
        return x

class AttLayer(nn.Module):
    def __init__(self, embedding_dim, n_head, fast_att):
        super(AttLayer, self).__init__()
        self.att1 = FastMultiHeadAttention(embedding_dim, n_head, fast_att)
        self.conv1 = ConvBloc1d(embedding_dim, embedding_dim)
                  
    def forward(self, x, mask, dist):
        nodes = x
        nodes = self.att1(nodes, nodes, mask, dist)     
        nodes = self.conv1(nodes)
        return nodes

def metric(y_pred, y_true):
    return np.mean(np.abs(y_true.ravel() - y_pred.ravel()))

class Model(nn.Module):
    def __init__(self, CONFIG, loss=False): 
        super(Model, self).__init__()
        self.hidden_size = CONFIG['hidden_size']
        self.num_head = CONFIG['n_head']
        self.num_layer = CONFIG['num_layer']
        self.fast_att = CONFIG['fast_att']
        self.mol = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.atom_encoder = AtomEncoder(emb_dim=self.hidden_size) # Pytorch Module class w/ learnable parameters
        self.degree_encoder = nn.Embedding(CONFIG['max_degree']+1, self.hidden_size) 
        glorot_uniform(self.degree_encoder.weight)
        self.bond_encoder = BondEncoder(emb_dim=self.hidden_size) # Pytorch Module class w/ learnable parameters  
        self.ring_encoder = nn.Embedding(CONFIG['max_ring_length']+1, self.hidden_size) 
        glorot_uniform(self.ring_encoder.weight)
        self.backbone = nn.ModuleList([AttLayer(self.hidden_size, self.num_head, self.fast_att,)
                                               for i in range(self.num_layer)])
        self.fc = nn.Linear(self.hidden_size, 1)
        self.target_mean = CONFIG['target_mean']
        self.loss = loss
        
    def forward(self, input_dict):
        node_feat = input_dict['node_feat']
        edge_mask = input_dict['edge_mask']
        num_nodes = input_dict['num_nodes']
        num_edges = input_dict['num_edges']
        if self.fast_att:
            edge_mask = self.fast_att * (1 - edge_mask)
        x = self.atom_encoder(node_feat) # bs x num_nodes x hidden_size
        degrees = input_dict['degrees']
        x_degree = self.degree_encoder(degrees)
        x = x + x_degree
        bs, num_max_nodes, _ = x.shape
        edge_feat = input_dict['edge_feat']
        xe = self.bond_encoder(edge_feat) # bs x num_edges x hidden_size
        bs, num_max_edges, _ = xe.shape
        ring_lengths = input_dict['ring_lengths']
        num_rings = input_dict['num_rings']
        xr = self.ring_encoder(ring_lengths) # bs x num_rings x hidden_size
        bs, num_max_rings, _ = xr.shape
        xs = [torch.cat((x[i, :num_nodes[i]], 
                         xe[i, :num_edges[i]],
                         xr[i, :num_rings[i]],
                        )) 
              for i in range(bs)]
        num_max = max(xs[i].shape[0] for i in range(bs))
        xs = [F.pad(xs[i], ( 0, 0, 0, num_max - xs[i].shape[0],)) 
              for i in range(bs)]
        x = torch.stack(xs)
        mol = self.mol.expand(bs, -1, -1)
        mol = mol + 1. / (num_nodes.unsqueeze(-1) + 1e-2)
        x = torch.cat([mol, x], 1)
        edge_mask = edge_mask[:, :x.shape[1], :x.shape[1]] # for multi gpu x resizing 
            
        for att_layer in self.backbone:
            x = att_layer(x, edge_mask, 0)
        x0 = x[:, 0]
        preds = self.fc(x0) + self.target_mean
        logits = None
        output_dict = {
            'preds' : preds,
        }
        if self.loss:
            target = input_dict['target']
            criterion = nn.L1Loss()
            loss = criterion(preds, target)
            output_dict['loss'] = loss                
                
        return output_dict

def train_epoch(loader, model, optimizer, scheduler, scaler, device, CONFIG):
    model.train()
    model.zero_grad()
    losses = []
    if CONFIG['verbose']:
        bar = tqdm(range(len(loader)))
    else:
        bar = range(len(loader))
    load_iter = iter(loader)
    preds = []
    targets = []
    #accumulate = 0
    
    for i, batch in zip(bar, load_iter):
        
        input_dict = {k:v.to(device, non_blocking=True) for k,v in batch.items()}
            
        with autocast():
            out_dict = model(input_dict)
            loss = out_dict['loss']
            if CONFIG['dp']:
                loss = loss.mean()
                
            loss = loss / (CONFIG['n_accumulate'])
            loss_np = loss.detach().cpu().numpy()
            
        scaler.scale(loss).backward()
        
        if True:
            if CONFIG['grad_norm'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_norm'])
            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            accumulate = 0
        scheduler.step()       
        losses.append(loss_np)
        smooth_loss = np.mean(losses) 
        
        if CONFIG['verbose']:
            bar.set_description('loss: %.4f, smth: %.4f, ' % (loss_np, smooth_loss, ))
        
        preds.append(out_dict['preds'].detach().cpu())
        targets.append(input_dict['target'].detach().cpu())
        

    preds = torch.cat(preds, 0).numpy()    
    targets = torch.cat(targets, 0).numpy()      
    
    score = metric(preds, targets)
    return np.mean(losses), score, preds, targets

def val_epoch(loader, model, device, CONFIG):
    model.eval()
    model.zero_grad()
    losses = []
    if CONFIG['verbose']:
        bar = tqdm(range(len(loader)))
    else:
        bar = range(len(loader))
    load_iter = iter(loader)
    preds = []
    targets = []
    #accumulate = 0
    
    with torch.no_grad():
        for i, batch in zip(bar, load_iter):

            input_dict = {k:v.to(device, non_blocking=True) for k,v in batch.items()}

            with autocast():
                out_dict = model(input_dict)
                loss = out_dict['loss']
                if CONFIG['dp']:
                    loss = loss.mean()
                loss = loss / (CONFIG['n_accumulate'])
                loss_np = loss.detach().cpu().numpy()

            losses.append(loss_np)
            smooth_loss = np.mean(losses) 

            if CONFIG['verbose']:
                bar.set_description('loss: %.4f, smth: %.4f, ' % (loss_np, smooth_loss, ))

            preds.append(out_dict['preds'].detach().cpu())
            targets.append(input_dict['target'].detach().cpu())
        

    preds = torch.cat(preds, 0).numpy()      
    targets = torch.cat(targets, 0).numpy()      
    
    score = metric(preds, targets)
    return np.mean(losses), score, preds, targets

def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0, head_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "backbone" not in n],
             'lr': decoder_lr, 'weight_decay': head_decay}
        ]
        return optimizer_parameters

def get_optimizer(model, CONFIG):
    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CONFIG['learning_rate'], 
                                                decoder_lr=CONFIG['head_learning_rate'],
                                                weight_decay=CONFIG['weight_decay'],
                                                head_decay=CONFIG['head_decay'],
                                               )
    optimizer = torch.optim.AdamW(optimizer_parameters, 
                      betas=(CONFIG['opt_beta1'], CONFIG['opt_beta2']),
                      eps=CONFIG['opt_eps'],
                      lr = CONFIG['learning_rate'],
                     )
    return optimizer

def get_scheduler(optimizer, train_data_loader, CONFIG):
    if CONFIG['scheduler'] == 'SineAnnealingLR':
        T_max = int(np.ceil(0.5*len(train_data_loader) * CONFIG['epochs']))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=T_max, 
                                                   eta_min=CONFIG['min_lr'])
        for t in range(T_max):
            scheduler.step()
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'cosine':
        num_train_steps = int(len(train_data_loader) * CONFIG['epochs'])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=CONFIG['num_warmup_steps'], num_training_steps=num_train_steps, num_cycles=CONFIG['num_cycles']
        )
        return scheduler

    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler

def save_checkpoint(model, epoch, fold, fname, output_dir):
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epoch,
        'fold':fold,
        }
    torch.save(checkpoint, '%s/%s/%s_%d.pt' % (output_dir, fname, fname, fold))

def train(CONFIG, dataset, train_idx, valid_idx, output_dir):
    fname = CONFIG['fname']
    fold = CONFIG['fold']
    
    print(fname)
    device = torch.device('cuda')
    scores = []

    seed_torch(CONFIG['seed'])

    model = Model(CONFIG, loss=True).to(device)
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print(total_params)

    train_data_loader = get_data_loader(dataset, train_idx, shuffle=True, CONFIG=CONFIG)
    valid_data_loader = get_data_loader(dataset, valid_idx, shuffle=False, CONFIG=CONFIG)

    optimizer = get_optimizer(model, CONFIG)
    scheduler = get_scheduler(optimizer, train_data_loader, CONFIG)
    scaler = GradScaler()
    if CONFIG['dp']: 
        model = nn.DataParallel(model)

    start_epoch = 0
    best_score = 10000
    for epoch in range(start_epoch, CONFIG['epochs']):
        print(time.ctime(), 'Epoch:', epoch, flush=True)
        train_loss, train_score, _, _ = train_epoch(train_data_loader, model, 
                                                    optimizer, scheduler, scaler, 
                                                    device, CONFIG)        
        val_loss, val_score, _ , _ = val_epoch(valid_data_loader, model, 
                                               device, CONFIG)                           
        content = 'Fold %d Ep %d train loss: %.5f metric: %.5f val loss: %.5f metric: %5f'
        values = (fold, epoch, train_loss, train_score, val_loss, val_score)
        print(content % values, flush=True)
        score = val_score
        if val_score < best_score:
            print('score improved from %0.4f to %0.4f' % (best_score, score))
            best_score = val_score
            if CONFIG['dp']:
                save_checkpoint(model.module, epoch, fold, fname, output_dir)
            else:
                save_checkpoint(model, epoch, fold, fname, output_dir)

    scores.append(best_score)
    del  valid_data_loader
    del train_data_loader, model, optimizer, scheduler, scaler
    gc.collect()

def run(fname, fold, input_dir, output_dir, split_path, cuda_devices):
            
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices

    fname = 'cpmp_final'
    
    CONFIG = init_config(cuda_devices, fname, fold, output_dir)
    seed_torch(CONFIG['seed'])

    dataset = get_data(input_dir)
    train_idx, valid_idx  = get_split(dataset, split_path, CONFIG)
    train(CONFIG, dataset, train_idx, valid_idx, output_dir)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fname", help='checkpoint dir name to be used')
    parser.add_argument("--fold", help='train fold to be used')
    parser.add_argument("--input_dir", help='directory for ogb data')
    parser.add_argument("--output_dir", help='checkpoints directory')
    parser.add_argument("--cuda_devices", help='visible cuda devices')
    parser.add_argument("--split_path", help='split path')
    
    args = parser.parse_args()
    
    if args.fname:
        fname = args.fname
    else:
        fname = 'cpmp_test'
    print('fname', fname)                 
        
    if args.fold:
        fold = int(args.fold)
    else:
        fold = None
    print('fold', fold)
    
    if args.input_dir:
        input_dir = args.input_dir
    else:
        input_dir = '/raid/pcqm4mv2ring'
    print('input dir', input_dir)
        
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = '../checkpoints'
    print('output_dir', output_dir)
        
    if args.cuda_devices:
        cuda_devices = args.cuda_devices
    else:
        cuda_devices = "0"
    print('cuda visible devices', cuda_devices)
    
    if args.split_path:
        split_path = args.split_path
    else:
        split_path = '../input/new_split_dict.pt'
    print('split path', split_path)

    run(fname, fold, input_dir, output_dir, split_path, cuda_devices)
    