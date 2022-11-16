from torch import nn
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter
import torch

def build_module_list(hs):
    w = nn.ModuleList()
    for c,h in enumerate(hs[:-1]):
        w.append(nn.Linear(h, hs[c+1]))
        if c != len(hs)-2:
            w.append(nn.ReLU())
    return w

def list_forward(w, x):
    for l in w:
        x = l(x)
    return x

class PointDensePooling(nn.Module):
    def __init__(self, hs):
        super(PointDensePooling, self).__init__()
        self.w = build_module_list(hs+[1])
        
    def forward(self, x, batch_index):
        a = list_forward(self.w, x)
        a = torch.exp(a)
        x = global_add_pool(x*a, batch_index)
        a = global_add_pool(a, batch_index)
        return x/a
    
