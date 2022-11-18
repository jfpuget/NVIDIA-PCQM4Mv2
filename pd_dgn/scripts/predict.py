import pytorch_lightning as pl
import torch.nn.functional as F
from tqdm import tqdm
import utils
from constants import PATH,MAX_NODE_FEA,PRED_PATH
from datasets import get_dl,gety
from gnn import DGCN
from ogb.lsc import PCQM4Mv2Evaluator
import torch
from glob import glob
import numpy as np
from pathlib import Path

def gnn_predict(quick_run, yaml_path):

    config = utils.load_yaml(yaml_path)
    print(config)

    folds = 4 
    scores = [0.0846,0.0841,0.0841,0.0841]
    y1,ys = [],[]
    for c in range(folds):
        train_dl,valid_dl,testdev_dl,testchallenge_dl,num_feas = get_dl(PATH,
                                                                        config,use_kfold=True,
                                                                        fold=c,quick_run=quick_run)
        print(f'train: {len(train_dl)} valid: {len(valid_dl)}')
        print(f'testdev: {len(testdev_dl)} testchallenge: {len(testchallenge_dl)}')

        evaluator = PCQM4Mv2Evaluator()
        ckpts = glob(f'{PATH}/fold_{c}/version_0/checkpoints/*.ckpt')
        print(ckpts)
        assert len(ckpts) == 1
        ckpt = ckpts[0]
        
        model = DGCN(MAX_NODE_FEA, num_feas, config)
        
        trainer = pl.Trainer(gpus=1)
        y = gety(valid_dl)
        ys.append(y)

        print('Predict with {ckpt}')
        yp_valid = trainer.predict(model, valid_dl, ckpt_path=ckpt)
        yp_valid = torch.cat(yp_valid,dim=0)
        y1.append(yp_valid)

        input_dict = {'y_pred': yp_valid, 'y_true': y}
        result_dict = evaluator.eval(input_dict)
        score2 = result_dict['mae']
        print(f'Valid MAE: {score2:.4f}')
        

        yp_testdev = trainer.predict(model, testdev_dl, ckpt_path=ckpt)
        yp_testdev = torch.cat(yp_testdev,dim=0)

        yp_testchallenge = trainer.predict(model, testchallenge_dl, ckpt_path=ckpt)
        yp_testchallenge = torch.cat(yp_testchallenge,dim=0)
        
        save_path = f'{PRED_PATH}_fold{c}_valid'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        np.save(f'{save_path}/valid.npy', yp_valid.detach().cpu().float().numpy())
        np.save(f'{save_path}/testdev.npy', yp_testdev.detach().cpu().float().numpy())
        np.save(f'{save_path}/testchallenge.npy', yp_testchallenge.detach().cpu().float().numpy())


    yp = torch.cat(y1)
    y = torch.cat(ys)
    input_dict = {'y_pred': yp, 'y_true': y}
    result_dict = evaluator.eval(input_dict)
    score2 = result_dict['mae']
    print(f'Ave Valid MAE: {score2:.4f}')


if __name__ == "__main__":
    quick_run = False 
    yaml_path=f'../yaml/gnn.yaml'
    gnn_predict(quick_run=quick_run,yaml_path=yaml_path)
