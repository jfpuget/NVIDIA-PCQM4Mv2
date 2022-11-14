import argparse
parser = argparse.ArgumentParser(description='Traing gnn')
parser.add_argument('--gpu','-g',dest='gpu',default=0)
parser.add_argument('--seed','-s',dest='seed',default=0,type=int)
parser.add_argument('--fold','-f',dest='fold',default=0,type=int)
args = parser.parse_args()
print(args)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import pytorch_lightning as pl
pl.utilities.seed.seed_everything(seed=args.seed, workers=True)
import torch.nn.functional as F
from tqdm import tqdm
import utils
from constants import PATH,MAX_NODE_FEA
from datasets import get_dl,gety
from gnn import DGCN
from pytorch_lightning.callbacks import TQDMProgressBar
from ogb.lsc import PCQM4Mv2Evaluator
import torch

def gnn_train(quick_run=False, logger=True, yaml_path='../yaml/gnn.yaml',
          config=None):

    if config is None:
        config = utils.load_yaml(yaml_path)
    print(config)

    train_dl, valid_dl, _, _, num_feas = get_dl(PATH,config,use_kfold=config.use_kfold,fold=args.fold,quick_run=quick_run)
    print(f'train: {len(train_dl)} valid: {len(valid_dl)}')

    model = DGCN(MAX_NODE_FEA, num_feas, config)

    print('Start training:')
    EPOCHS = 1 if quick_run else config.epochs
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='valid_mae', mode='min')
    pcb = TQDMProgressBar(refresh_rate=20)
    trainer = pl.Trainer(gpus=1, max_epochs=EPOCHS, 
                     callbacks=[checkpoint_callback,pcb],
                     logger=logger,
                     precision=16,
                     gradient_clip_val=config.gradient_clip_val,
                     gradient_clip_algorithm="value"
                    )

    trainer.fit(model, train_dataloaders=train_dl, 
                val_dataloaders=valid_dl)

    print('Predict valid:')
    yp = trainer.predict(model, valid_dl)
    yp = torch.cat(yp,dim=0)
    y = gety(valid_dl)

    score1 = F.l1_loss(yp, y).item()

    evaluator = PCQM4Mv2Evaluator()
    input_dict = {'y_pred': yp, 'y_true': y}
    result_dict = evaluator.eval(input_dict)
    score2 = result_dict['mae']

    msgs = {'VALID MAE':score2}
    print(f'Valid MAE: {score1:.4f} {score2:.4f}')
    print("weights saved:", trainer.log_dir)
    return msgs


if __name__ == "__main__":
    quick_run = False 
    yaml_path='../yaml/gnn.yaml'
    gnn_train(quick_run=quick_run,yaml_path=yaml_path)
