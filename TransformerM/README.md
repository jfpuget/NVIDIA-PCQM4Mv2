# Transformer-M

## Environment Setup

Build and run the container using [docker](https://www.docker.com/)

```bash
# build the container
docker build -t transformerm .
mkdir results data
# copy custom folds
cp ../data/new_split_dict.pt data/
# run the container interactively
docker run -it --gpus=all --ipc=host --rm -v $(pwd)/../ensemble/models_oofs/predictions:/logs -v $(pwd)/results:/results -v $(pwd)/data:/data transformerm
export FOLD_IDX=0
export FOLD_PATH=/data/new_split_dict.pt
```
## Dataset

By running the corresponding commands below, the dataset will be downloaded and preprocessed, if the data hasn't been preprocessed apriori at the `--data_dir` location.

For the case where `--dataset-name` is `pcqm4mv2-3d-descriptor` it may take up to 48 hours or more to preprocess the complete data. Instead you could download the corresponding files from [here](https://drive.google.com/drive/folders/15Fx4UzQN8KNUedmNMYmbd1YkF1K3zXZZ?usp=share_link) and set the `--data_dir` to the location where the files are downloaded to. For example if the files are extracted in `/workspace/data/` then `--data_dir` to this path with the following content:

```
/workspace/data/
├── pcqm4m-v2-3d-descriptor
│   ├── processed
│   │   ├── geometric_data_processed.pt
│   │   ├── pre_filter.pt
│   │   └── pre_transform.pt
│   ├── raw
│   │   ├── data.csv.gz
│   │   └── pcqm4m-v2-train.sdf
│   ├── RELEASE_v1.txt
│   └── split_dict.pt
└── pcqm-dpfp
    ├── descriptor.np
    └── fingerprint.np
```

## Training Commands

### $\textrm{Transformer-M}^\textrm{base}_\textrm{without\\_denoising}$

```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --max_restarts 0 --module graphormer.runtime.training \
--dataset-source ogb --dataset-name pcqm4mv2-3d \
--num-workers 8 \
--criterion l1_loss \
--metric mae \
--architecture base \
--num-classes 1 \
--attention-dropout 0.1 \
--act-dropout 0.1 \
--dropout 0.0 \
--drop-path-prob 0.0 \
--position-noise 0.2 \
--channel-prob-2d-only 0.25 \
--channel-prob-3d-only 0.25 \
--channel-prob-2d-3d 0.5 \
--gradient-clip 5.0 \
--weight-decay 0.0 \
--warmup-updates 60000 \
--learning-rate 2e-4 \
--end-learning-rate 1e-14 \
--batch-size 128 \
--amp \
--epochs 400 \
--ckpt-interval 20 \
--eval-interval 20 \
--save-ckpt-path /results/${NAME}_fold${FOLD_IDX}.pth \
--data-dir /data \
--log-dir /logs/${NAME}_fold${FOLD_IDX} \
--prediction-name valid.npy \
--cv-fold-path $FOLD_PATH \
--cv-fold-idx $FOLD_IDX \
--seed 42
```

Run it 4 times (one for each fold) with
```
NAME=alexandrem_tm18_nonodepred_400 FOLD_IDX=0
NAME=alexandrem_tm18_nonodepred_400 FOLD_IDX=1
NAME=alexandrem_tm18_nonodepred_400 FOLD_IDX=2
NAME=alexandrem_tm18_nonodepred_400 FOLD_IDX=3
```

### $\textrm{Transformer-M}^\textrm{large}_\textrm{with\\_denoising}$


```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --max_restarts 0 --module graphormer.runtime.training \
--dataset-source ogb --dataset-name pcqm4mv2-3d \
--num-workers 8 \
--criterion l1_loss atom_denoise_loss \
--node-pos-head \
--metric mae \
--architecture medium-768 \
--num-classes 1 \
--attention-dropout 0.1 \
--act-dropout 0.1 \
--dropout 0.0 \
--drop-path-prob 0.1 \
--position-noise 0.2 \
--channel-prob-2d-only 0.25 \
--channel-prob-3d-only 0.25 \
--channel-prob-2d-3d 0.5 \
--gradient-clip 5.0 \
--weight-decay 0.0 \
--warmup-updates 60000 \
--learning-rate 2e-4 \
--end-learning-rate 1e-14 \
--batch-size 128 \
--amp \
--epochs 400 \
--ckpt-interval 20 \
--eval-interval 20 \
--save-ckpt-path /results/${NAME}_fold${FOLD_IDX}.pth \
--data-dir /data \
--log-dir /logs/${NAME}_fold${FOLD_IDX} \
--prediction-name valid.npy \
--cv-fold-path $FOLD_PATH \
--cv-fold-idx $FOLD_IDX
```

Run it 4 times (one for each fold) with
```
NAME=alexandrem_tm18forreal_withnodepred_400_2 FOLD_IDX=0
NAME=alexandrem_tm18forreal_withnodepred_400_2 FOLD_IDX=1
NAME=alexandrem_tm18forreal_withnodepred_400_2 FOLD_IDX=2
NAME=alexandrem_tm18forreal_withnodepred_400_2 FOLD_IDX=3
```

### $\textrm{Transformer-M}^{\textrm{large}}_{\textrm{baseline}}$

```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --max_restarts 0 --module graphormer.runtime.training \
--dataset-source ogb --dataset-name pcqm4mv2-3d \
--num-workers 8 \
--criterion l1_loss atom_denoise_loss \
--node-pos-head \
--metric mae \
--architecture medium-768 \
--num-classes 1 \
--attention-dropout 0.1 \
--act-dropout 0.1 \
--dropout 0.0 \
--drop-path-prob 0.1 \
--position-noise 0.2 \
--channel-prob-2d-only 0.2 \
--channel-prob-3d-only 0.2 \
--channel-prob-2d-3d 0.6 \
--gradient-clip 5.0 \
--weight-decay 0.0 \
--warmup-updates 150000 \
--learning-rate 2e-4 \
--end-learning-rate 1e-9 \
--batch-size 128 \
--amp \
--epochs 454 \
--ckpt-interval 20 \
--eval-interval 20 \
--save-ckpt-path /results/${NAME}_fold${FOLD_IDX}.pth \
--data-dir /data \
--log-dir /logs/${NAME}_fold${FOLD_IDX} \
--prediction-name valid.npy \
--cv-fold-idx $FOLD_IDX \
--cv-fold-path $FOLD_PATH \
--seed $SEED
```

Run it 8 times (two for each fold) with
```
NAME=sajad_blv4_no_dihedral_18l_val SEED=42 FOLD_IDX=0
NAME=sajad_blv4_no_dihedral_18l_val SEED=42 FOLD_IDX=0
NAME=sajad_blv4_no_dihedral_18l_val SEED=42 FOLD_IDX=0
NAME=sajad_blv4_no_dihedral_18l_val SEED=42 FOLD_IDX=0

NAME=sajad_blv5__18l_454_3407_val SEED=3407 FOLD_IDX=0
NAME=sajad_blv5__18l_454_3407_val SEED=3407 FOLD_IDX=1
NAME=sajad_blv5__18l_454_3407_val SEED=3407 FOLD_IDX=2
NAME=sajad_blv5__18l_454_3407_val SEED=3407 FOLD_IDX=3
```


### $\textrm{Transformer-M}^{\textrm{large}}_{\textrm{Dirichlet}}$

```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --max_restarts 0 --module graphormer.runtime.training \
--dataset-source ogb --dataset-name pcqm4mv2-3d \
--num-workers 8 \
--criterion l1_loss atom_denoise_loss \
--node-pos-head \
--metric mae \
--architecture medium-768 \
--num-classes 1 \
--attention-dropout 0.1 \
--act-dropout 0.1 \
--dropout 0.0 \
--drop-path-prob 0.1 \
--position-noise 0.2 \
--random-tp \
--channel-prob-2d-only 0.2 \
--channel-prob-3d-only 0.2 \
--channel-prob-2d-3d 0.6 \
--gradient-clip 5.0 \
--weight-decay 0.0 \
--warmup-updates 150000 \
--learning-rate 2e-4 \
--end-learning-rate 1e-9 \
--batch-size 128 \
--amp \
--epochs 454 \
--ckpt-interval 20 \
--eval-interval 20 \
--save-ckpt-path /results/${NAME}_fold${FOLD_IDX}.pth \
--data-dir /data \
--log-dir /logs/${NAME}_fold${FOLD_IDX} \
--prediction-name valid.npy \
--cv-fold-idx $FOLD_IDX \
--cv-fold-path $FOLD_PATH \
--seed 3407
```

Run it 4 times (one for each fold) with

```
NAME=sajad_blv1_18l_454_3407_val FOLD_IDX=0
NAME=sajad_blv1_18l_454_3407_val FOLD_IDX=1
NAME=sajad_blv1_18l_454_3407_val FOLD_IDX=2
NAME=sajad_blv1_18l_454_3407_val FOLD_IDX=3
```

For this variant, it was also trained on the full train+valid. To do so, **remove** the arguments `--cv-fold-idx`, `--cv-fold-path`, and **add** `--full-train` argument to the above command, and run with the following environment:

```
NAME=sajad_blv1_full_train
```

### $\textrm{Transformer-M}_{\textrm{kpgt}}^{\textrm{large}}$

```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --max_restarts 0 --module graphormer.runtime.training \
--dataset-source ogb --dataset-name pcqm4mv2-3d-descriptor \
--num-workers 8 \
--criterion l1_loss atom_denoise_loss kpgt_loss \
--node-pos-head \
--metric mae \
--architecture medium-768 \
--num-classes 1 \
--attention-dropout 0.1 \
--act-dropout 0.1 \
--dropout 0.0 \
--drop-path-prob 0.1 \
--position-noise 0.2 \
--channel-prob-2d-only 0.2 \
--channel-prob-3d-only 0.2 \
--channel-prob-2d-3d 0.6 \
--gradient-clip 5.0 \
--weight-decay 0.0 \
--warmup-updates 150000 \
--learning-rate 2e-4 \
--end-learning-rate 1e-9 \
--batch-size 128 \
--amp \
--epochs 454 \
--ckpt-interval 20 \
--eval-interval 20 \
--save-ckpt-path /results/${NAME}_fold${FOLD_IDX}.pth \
--data-dir /data \
--log-dir /logs/${NAME}_fold${FOLD_IDX} \
--prediction-name valid.npy \
--cv-fold-idx $FOLD_IDX \
--cv-fold-path $FOLD_PATH \
--kpgt-loss-weight-fp $LAMBDA \
--kpgt-loss-weight-dc $LAMBDA \
--position-noise 0.2 \
--seed 3407
```

Run it 8 times (two for each fold) with

```
NAME=sajad_blv2_r_0.1_18l_454_3407_val LAMBDA=0.1 FOLD_IDX=0
NAME=sajad_blv2_r_0.1_18l_454_3407_val LAMBDA=0.1 FOLD_IDX=1
NAME=sajad_blv2_r_0.1_18l_454_3407_val LAMBDA=0.1 FOLD_IDX=2
NAME=sajad_blv2_r_0.1_18l_454_3407_val LAMBDA=0.1 FOLD_IDX=3

NAME=sajad_blv2-0.2_18l_val LAMBDA=0.2 FOLD_IDX=0
NAME=sajad_blv2-0.2_18l_val LAMBDA=0.2 FOLD_IDX=1
NAME=sajad_blv2-0.2_18l_val LAMBDA=0.2 FOLD_IDX=2
NAME=sajad_blv2-0.2_18l_val LAMBDA=0.2 FOLD_IDX=3
```

Two models were trained with $\lambda=0.1$ and $\lambda=0.2$ corresponding to the weights of the parameters `--kpgt-loss-weight-fp` and `--kpgt-loss-weight-dc`. For these variants, they were also trained on the full train+valid. To do so, **remove** the arguments `--cv-fold-idx`, `--cv-fold-path`, and **add** `--full-train` argument to the above command, and run with the following environment:

```
NAME=sajad_blv2_full_train LAMBDA=0.1
NAME=sajad_blv2-0.2_full_train LAMBDA=0.2
```

## Inference

To run inference, run the same command as for training, replacing `graphormer.runtime.training` with `graphormer.runtime.inference`, providing a `load-ckpt-path` argument, and setting `--prediction-name testchallenge.npy`.


## References

```
@article{2022transformerm,
  author = {Luo, Shengjie and Chen, Tianlang and Xu, Yixian and Zheng, Shuxin and Liu, Tie-Yan and He, Di and Wang, Liwei},
  title = {One Transformer Can Understand Both 2D &amp; 3D Molecular Data},
  publisher = {arXiv},
  year = {2022},
  url = {https://arxiv.org/abs/2210.01765},
}
```

```
@article{shi2022benchmarking,
  title={Benchmarking Graphormer on Large-Scale Molecular Modeling Datasets},
  author={Yu Shi and Shuxin Zheng and Guolin Ke and Yifei Shen and Jiacheng You and Jiyan He and Shengjie Luo and Chang Liu and Di He and Tie-Yan Liu},
  journal={arXiv preprint arXiv:2203.04810},
  year={2022},
  url={https://arxiv.org/abs/2203.04810}
}
```

```
@inproceedings{ying2021do,
    title={Do Transformers Really Perform Badly for Graph Representation?},
    author={Chengxuan Ying and Tianle Cai and Shengjie Luo and Shuxin Zheng and Guolin Ke and Di He and Yanming Shen and Tie-Yan Liu},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021},
    url={https://openreview.net/forum?id=OeWooOxFwDa}
}
```
