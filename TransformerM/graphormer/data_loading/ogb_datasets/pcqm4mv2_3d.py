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

import rdkit.Chem as Chem
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import os.path as osp
import shutil
import tarfile
import multiprocessing as mp

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from ogb.utils.features import atom_to_feature_vector
from ogb.utils.features import bond_to_feature_vector
from ogb.utils.url import download_url
from ogb.utils.url import extract_zip
import itertools

from .descriptors.rdNormalizedDescriptors import RDKit2DNormalized


def extract_tar_gz(filename, root):
    print('Extracting', filename)
    tar = tarfile.open(filename, 'r:gz')
    tar.extractall(path=root)


def mol2graph(mol):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
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
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    if mol.GetNumConformers():
        graph['node_pos'] = mol.GetConformer(0).GetPositions()

    return graph


class Pyg3DPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, root='dataset', use_sdf=True, compute_descriptors=False, transform=None, pre_transform=None):
        self.original_root = root
        self.folder = osp.join(root, 'pcqm4m-v2-3d-descriptor' if compute_descriptors else 'pcqm4m-v2-3d')
        self.version = 1
        self.use_sdf = use_sdf

        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'
        self.url_3d = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz'

        self.compute_descriptors = compute_descriptors
        # check version and update if necessary
        if osp.isdir(self.folder) and (
                not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.csv.gz', 'pcqm4m-v2-train.sdf']

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.folder)
            os.rename(osp.join(self.original_root, 'pcqm4m-v2'), self.folder)
            path_3d = download_url(self.url_3d, self.original_root)
            extract_tar_gz(path_3d, self.raw_dir)
            os.unlink(path_3d)
        else:
            print('Stop download.')
            exit(-1)

    def process_item(self, smile, sdf_mol, homolumogap, idx):
        data = Data()
        if self.use_sdf and sdf_mol:
            mol = sdf_mol
        else:
            mol = Chem.MolFromSmiles(smile)

        graph = mol2graph(mol)

        assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert (len(graph['node_feat']) == graph['num_nodes'])

        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        data.y = torch.Tensor([homolumogap])
        data.smile = smile

        if 'node_pos' in graph:
            data.node_pos = torch.from_numpy(graph['node_pos']).to(torch.float32)
        else:
            data.node_pos = torch.zeros(int(graph['num_nodes']), 3)

        if self.compute_descriptors:
            self.dc_memmap[idx] = np.asarray(RDKit2DNormalized().process(smile), dtype=np.float32)
            self.fp_memmap[idx] = np.asarray(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512), dtype=np.int64)
        return data

    def process(self):
        structures = Chem.SDMolSupplier(osp.join(self.raw_dir, 'pcqm4m-v2-train.sdf'),
                                        strictParsing=False)
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
       
        if self.compute_descriptors:
            fp_memmap = self.original_root / 'pcqm-dpfp' / 'fingerprint.np'
            dc_memmap = self.original_root / 'pcqm-dpfp' / 'descriptor.np'
            os.makedirs(self.original_root / 'pcqm-dpfp', exist_ok=True)
            self.fp_memmap = np.memmap(
                fp_memmap,
                dtype='float32',
                mode='w+',
                shape=(3746620, 512)
            )
            self.dc_memmap = np.memmap(
                dc_memmap,
                dtype='float32',
                mode='w+',
                shape=(3746620, 201)
            )
        print('Converting SMILES strings into graphs...')
        data_list = [self.process_item(*ll) for ll in tqdm(
            itertools.zip_longest(data_df['smiles'], structures, 
                                  data_df['homolumogap'], 
                                  np.arange(len(data_df.index))),
                                  total=len(data_df.index))]

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict
