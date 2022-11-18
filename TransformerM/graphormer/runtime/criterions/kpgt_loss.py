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

from pathlib import Path

import numpy
import torch
from torch import Tensor
from graphormer.runtime.criterions.metric import Metric

FINGERPRINT_POS_WEIGHTS = torch.tensor([1.1433, 0.8838, 0.3456, 0.5512, 1.6746, 0.9952, 1.5817, 1.6147, 1.4659,
                                        0.7606, 0.8977, 1.1196, 1.2970, 0.8610, 0.8582, 0.9803, 1.2453, 0.6388,
                                        1.3811, 1.5134, 1.1209, 1.5815, 1.0545, 1.0528, 0.9344, 0.9676, 1.2979,
                                        0.6696, 0.8143, 1.2438, 1.6053, 1.8365, 1.2509, 1.3739, 0.8337, 0.8341,
                                        0.2796, 1.1925, 0.8285, 1.5505, 0.4639, 0.7095, 1.2747, 1.1854, 0.9252,
                                        1.5336, 0.7029, 1.1529, 1.3404, 0.9466, 0.2867, 0.4112, 1.1681, 1.4228,
                                        1.7150, 1.3663, 0.6254, 1.1855, 1.6361, 1.2994, 1.3474, 0.7125, 1.4688,
                                        1.0894, 0.6387, 1.4126, 1.4762, 0.8569, 1.1310, 1.0335, 1.2628, 0.8903,
                                        0.8879, 1.0656, 1.8012, 1.2008, 0.9526, 1.1099, 1.6822, 1.0750, 1.5623,
                                        0.1904, 1.2061, 1.0030, 1.1734, 1.7299, 1.2252, 1.3098, 1.5960, 0.9591,
                                        1.0727, 0.9963, 1.1236, 0.8538, 0.6767, 1.0342, 0.8315, 1.1265, 0.9035,
                                        1.3957, 1.2200, 1.1377, 1.2433, 0.4589, 0.8421, 0.9691, 1.5336, 1.1061,
                                        1.2195, 0.9105, 1.0625, 1.0539, 0.4024, 0.9766, 1.2658, 1.2650, 1.0543,
                                        0.6457, 1.0505, 1.2629, 0.7788, 0.5350, 0.9544, 0.6437, 1.0530, 1.2013,
                                        1.0292, 1.4030, 1.2501, 1.0277, 1.2316, 1.6096, 0.7074, 1.8162, 2.0429,
                                        0.7929, 1.0074, 1.8915, 0.8902, 0.7904, 0.7646, 1.3226, 0.3332, 1.4240,
                                        0.5547, 0.7867, 1.0248, 1.1207, 0.5303, 0.7960, 1.1485, 0.9524, 1.2563,
                                        0.4450, 1.3372, 1.1377, 0.6708, 1.2597, 0.8595, 0.3687, 1.1885, 0.3796,
                                        0.7311, 0.8844, 1.8360, 1.4988, 0.5365, 0.7383, 1.0817, 1.3728, 0.5171,
                                        1.2147, 0.8348, 0.5612, 1.4485, 1.3215, 0.9140, 1.2246, 1.7293, 2.0300,
                                        0.8299, 0.8327, 0.7828, 1.3863, 1.4372, 1.2622, 0.9348, 1.7782, 1.4634,
                                        1.4101, 1.2950, 0.8984, 0.2338, 1.0134, 0.5407, 0.6542, 1.0422, 0.1298,
                                        1.0293, 1.2797, 0.8614, 1.2999, 0.8972, 1.4428, 1.1881, 0.9351, 0.7684,
                                        1.4831, 0.9783, 0.2466, 0.8616, 1.0074, 1.5260, 1.2573, 1.1077, 1.0518,
                                        1.1240, 0.8515, 1.2113, 0.8773, 1.5850, 1.2854, 0.9498, 0.9987, 1.3147,
                                        0.9659, 1.6144, 0.8407, 1.9652, 0.9832, 1.3266, 0.9564, 0.6291, 1.7470,
                                        1.0502, 0.6871, 0.0554, 0.7602, 0.6941, 1.8213, 1.2240, 1.2870, 0.5563,
                                        1.3823, 1.3920, 1.2942, 0.7316, 1.1846, 0.3475, 1.2789, 1.3426, 1.1278,
                                        1.0115, 0.6980, 0.9426, 0.7267, 0.7221, 1.7171, 1.1957, 1.3581, 1.3267,
                                        1.1763, 1.7853, 0.5873, 1.0405, 1.8795, 1.4466, 1.3858, 0.6688, 0.4198,
                                        0.6811, 1.3068, 1.1390, 1.0927, 0.6114, 1.3618, 0.6887, 0.1376, 0.8475,
                                        0.5827, 0.4763, 0.5215, 1.1553, 0.6325, 0.0464, 1.1584, 0.7460, 1.5230,
                                        1.2583, 1.4986, 1.4440, 0.5253, 1.3385, 1.6598, 0.4422, 1.1920, 0.8484,
                                        0.6452, 1.4779, 1.1515, 0.6439, 0.8346, 0.9369, 1.1015, 2.0083, 0.7788,
                                        0.8455, 1.5934, 0.5886, 0.6741, 1.6645, 1.0083, 0.5657, 0.5695, 0.7023,
                                        0.2980, 1.8634, 0.9990, 0.5345, 0.9419, 0.8232, 1.2976, 1.3019, 1.7212,
                                        1.1673, 1.0305, 0.9182, 1.1502, 1.3046, 0.6761, 0.4874, 0.8643, 1.5993,
                                        0.8762, 1.1904, 0.9288, 0.4799, 0.7687, 0.9641, 0.9572, 1.4493, 1.5708,
                                        0.7473, 0.5538, 1.4584, 0.9611, 0.9187, 0.8137, 1.7298, 1.0161, 1.0758,
                                        0.8683, 0.7193, 0.7371, 1.1493, 1.0508, 1.3558, 1.2689, 1.2407, 1.4628,
                                        0.9439, 1.1472, 1.5789, 0.3356, 1.0757, 1.4473, 0.9666, 1.0471, 1.3278,
                                        0.9079, 1.1866, 0.3489, 1.0090, 0.4928, 1.2187, 1.1505, 1.0002, 0.8380,
                                        1.4186, 0.9758, 1.4092, 0.4416, 0.8870, 0.9528, 0.9568, 1.2754, 0.5976,
                                        0.9177, 1.4636, 1.4043, 0.7567, 0.3972, 0.9438, 1.2843, 0.8397, 1.1775,
                                        1.0930, 0.8736, 1.6357, 1.1340, 1.3809, 1.5761, 1.3036, 0.8920, 1.6341,
                                        0.4455, 1.2860, 0.9360, 1.1623, 0.9389, 0.9001, 1.0110, 0.4272, 1.0167,
                                        1.6612, 0.9328, 1.5806, 0.6138, 0.3222, 1.5861, 0.7756, 1.0864, 1.7358,
                                        0.8059, 0.8061, 1.4492, 0.8507, 0.7798, 1.0128, 1.1744, 0.8980, 1.1273,
                                        0.5334, 1.1910, 1.0023, 0.9673, 1.2829, 1.3860, 1.2031, 0.8894, 0.5574,
                                        0.5035, 0.8839, 1.2628, 1.0772, 0.5374, 0.3721, 1.4517, 1.5993, 1.0078,
                                        1.2926, 1.3408, 1.7897, 1.2813, 1.4673, 0.7928, 1.3212, 1.6255, 1.6306,
                                        1.0476, 1.0144, 0.9799, 0.9052, 0.8052, 0.7397, 0.5368, 1.1788, 1.3305,
                                        1.2913, 1.2915, 0.7378, 0.9106, 0.9419, 1.2807, 1.2306, 1.2919, 1.3048,
                                        1.5725, 1.2672, 0.7778, 1.8494, 0.6516, 1.7830, 1.2120, 1.2058, 0.7410,
                                        1.7013, 1.0385, 1.6720, 1.5717, 1.3509, 0.8370, 1.3128, 1.5688, 1.2583,
                                        0.8904, 1.6605, 1.2982, 1.2291, 0.8997, 1.1351, 0.3735, 1.0625, 1.4635,
                                        1.6029, 0.8452, 0.8462, 1.4721, 0.9323, 1.4820, 1.3061, 0.2719])

class KPGTPretextLoss(Metric):
    def __init__(self, 
                 key='kpgt',
                 kpgt_loss_weight_dc=0.1,
                 kpgt_loss_weight_fp=0.1,
                 **kwargs):
        super().__init__('kpgt_loss')
        self.key = key
        self.add_state('error', torch.tensor(0, dtype=torch.float32, device='cuda'))
        self.add_state('total', torch.tensor(0, dtype=torch.int32, device='cuda'))
        self.orig_vals = None

        self.amp = kwargs['amp']
        self.data_dir = Path(kwargs['data_dir'])
        self.fp_memmap = None
        self.dc_memmap = None

        fp_memmap = self.data_dir / 'pcqm-dpfp' / 'fingerprint.np'
        dc_memmap = self.data_dir / 'pcqm-dpfp' / 'descriptor.np'
        self.kpgt_dc_criterion = torch.nn.MSELoss()
        self.kpgt_fp_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=FINGERPRINT_POS_WEIGHTS.to('cuda'))

        self.kpgt_loss_weight_dc = kpgt_loss_weight_dc
        self.kpgt_loss_weight_fp = kpgt_loss_weight_fp
        self.fp_memmap = numpy.array(numpy.memmap(
            fp_memmap,
            dtype='float32',
            mode='r',
            shape=(3746620, 512)
        ))

        self.dc_memmap = numpy.array(numpy.memmap(
            dc_memmap,
            dtype='float32',
            mode='r',
            shape=(3746620, 201)
        ))

    def update(self, 
               kpgt: Tensor, 
               targets: Tensor, 
               idx: Tensor, **_):
        with torch.cuda.amp.autocast(enabled=self.amp):
            logits_dc, logits_fp = kpgt
            n = logits_fp.shape[0]

            targets_dc = torch.from_numpy(numpy.array(
                self.dc_memmap[idx.cpu().numpy(), 1:])).to(logits_dc.device)

            targets_fp = torch.from_numpy(numpy.array(
                self.fp_memmap[idx.cpu().numpy(), :])).to(logits_fp.device)

            loss_fp = self.kpgt_fp_criterion(logits_fp, targets_fp)
            nan_mask = torch.isnan(targets_dc)
            logits_dc = torch.where(nan_mask, 0, logits_dc)
            targets_dc = torch.where(nan_mask, 0, targets_dc)
            loss_dc = self.kpgt_dc_criterion(logits_dc, targets_dc)
            loss_kpgt = loss_fp * self.kpgt_loss_weight_fp + loss_dc * self.kpgt_loss_weight_dc

        error = loss_kpgt
        self.total += n
        self.error += error.detach()
        return loss_kpgt

    def _compute(self):
        return self.error / self.total, {}
