# Transformer-M

## Build and run the container

With the latest docker with GPU support:

```bash
# build the container
docker build -t transformerm .
mkdir results data
# copy custom folds
cp ../new_split_dict.pt data/
# run the container interactively
docker run -it --gpus=all --ipc=host --rm -v ${PWD}/results:/results -v ${PWD}/data:/data transformerm
```

## Training

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
--ckpt-interval 10 \
--eval-interval 10 \
--save-ckpt-path /results/checkpoint_pcqm4mv2.pth \
--data_dir /data \
--log-dir /results
--cv-fold-path /data/new_split_dict.pt
--cv-fold-idx 0
--wandb
```

### $\textrm{Transformer-M}^\textrm{large}_\textrm{with\\_denoising}$


```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --max_restarts 0 --module graphormer.runtime.training \
--dataset-source ogb --dataset-name pcqm4mv2-3d \
--num-workers 8 \
--criterion l1_loss \
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
--ckpt-interval 10 \
--eval-interval 10 \
--save-ckpt-path /results/checkpoint_pcqm4mv2.pth \
--data_dir /data \
--log-dir /results
--cv-fold-path /data/new_split_dict.pt
--cv-fold-idx 0
--wandb
```
### sajad v2

### sajad v2-0.2

### sajad v4

### sajad v5


## Inference

To run inference, run the same command as for training, replacing `graphormer.runtime.training` with `graphormer.runtime.inference` and providing a `load-ckpt-path` argument.


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
