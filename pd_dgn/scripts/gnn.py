import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Sequential, global_add_pool, global_mean_pool, DeepGCNLayer
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv as Glayer
from pooling import PointDensePooling 
from constants import YMIN,YMAX
from torch.nn import Dropout, Linear, ReLU
import torch.nn.functional as F
import torch
import utils

class DGCN(pl.LightningModule):

    def __init__(self, max_node_fea, num_feas, config):
        super(DGCN, self).__init__()

        self.num_feas = num_feas
        self.lr = config.lr
        self.wd = config.wd
        self.epochs = config.epochs

        # hidden layer node features
        self.hidden = config.H1
        self.En = config.En
        Ee = config.Ee
        self.max_node_fea = max_node_fea + 1
        self.emb = torch.nn.Embedding(self.max_node_fea, self.En)
        heads = config.heads
        H = self.hidden // heads

        act = None if config.act == 'None' else config.act
        act = eval(config.act)
        self.l1 = Glayer(self.num_feas*self.En, H, heads, edge_dim=3, act=act, fill_value=1.0, beta=config.beta)
        ls = []
        for i in range(config.layers):
            ls.append(DeepGCNLayer(Glayer(H*heads, H, heads, edge_dim=3, act=act, fill_value=1.0, beta=config.beta)))
        self.layers = torch.nn.ModuleList(ls)
        self.p = PointDensePooling([H*heads, H*heads//2, H*heads//4])
        self.out = torch.nn.Sequential(Linear(H*heads, 1),
                                       Linear(1, 1))      

    def forward(self, x, edge_index, edge_attr, batch_index):
        x = self.emb(x).view(-1,self.num_feas*self.En)
        x = self.l1(x,edge_index,edge_attr.float()/4.0)
        x = F.relu(x)
        x_n = 0
        w = 0
        for i,l in enumerate(self.layers):
            x = l(x,edge_index,edge_attr.float()/4.0)
            x = F.relu(x)
            if i >= 10:
                x_n += self.p(x, batch_index)*i
                w += i
        x = x_n / w
        x1 = self.out(x)
        x2 = torch.clip(x1,YMIN,YMAX)
        x_out = (x1+x2)/2
        return x_out.squeeze()
    
    def _f(self, batch, batch_index):
        x, edge_index = batch.x, batch.edge_index
        edge_attr = batch.edge_attr
        batch_index = batch.batch
        x_out = self.forward(x, edge_index, edge_attr, batch_index)
        return x_out
    
    def _loss(self, batch, batch_index, tag):
        x_out = self._f(batch, batch_index)
        loss = F.smooth_l1_loss(x_out, batch.y, beta=0.1)
        x_out = torch.clip(x_out,YMIN,YMAX)
        mae = F.l1_loss(x_out, batch.y)
        self.log(f"{tag}_mae", mae, batch_size = batch.y.shape[0], prog_bar=True)
        return loss

    def training_step(self, batch, batch_index):
        return self._loss(batch, batch_index, 'train')

    def validation_step(self, batch, batch_index):
        return self._loss(batch, batch_index, 'valid')
        
    def predict_step(self, batch, batch_index):
        x_out = self._f(batch, batch_index)
        return torch.clip(x_out,YMIN,YMAX)

    def configure_optimizers(self):
        adam = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        slr = torch.optim.lr_scheduler.CosineAnnealingLR(adam, self.epochs)
        return [adam], [slr]

if __name__ == "__main__":
    config = utils.load_yaml('../yaml/gnn.yaml')
    model = DGCN(36,9,config)
    # 5 layers: 0.2008 val mae
    # 10 layers: 0.172 val mae
    # 10 lyaers, smooth l1 loss beta=0.1: 0.169
