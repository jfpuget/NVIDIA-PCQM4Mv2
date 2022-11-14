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

class NumAtom(torch.nn.Module):
    def __init__(self, hs):
        super(NumAtom, self).__init__()
        self.w = build_module_list([1]+hs)
        
    def forward(self, x, batch_index):
        x = torch.ones(x.size()[0],1).to(x.device)
        x = global_add_pool(x, batch_index)
        x = 1.0/x
        x = list_forward(self.w, x)
        return x

class PointAtt(nn.Module):
    def __init__(self, hs):
        super(PointAtt, self).__init__()
        self.w = build_module_list(hs+[1])
        
    def forward(self, x, batch_index):
        a = list_forward(self.w, x)
        a = torch.exp(a)
        x = global_add_pool(x*a, batch_index)
        a = global_add_pool(a, batch_index)
        return x/a
    
class SelfAtt(nn.Module):
    def __init__(self, hs):
        super(SelfAtt, self).__init__()
        self.q = build_module_list(hs)
        self.k = build_module_list(hs)
        
    def forward(self, x, batch_index):
        q = list_forward(self.q, x)
        k = list_forward(self.k, x)

        a = torch.tensordot(q,k,dims=([1], [1]))/k.size()[1]
        a = torch.exp(a)
        a = scatter(a,batch_index)
        rid = torch.arange(a.size()[0])
        a = a[rid, batch_index]
        a = a.view(-1,1)
        x = global_add_pool(x*a, batch_index)
        a = global_add_pool(a, batch_index)
        return x/(a+1e-6)
