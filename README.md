# OverlapNetVLAD

This repository represents the official implementation of the paper:

**OverlapNetVLAD: A Coarse-to-Fine Framework for LiDAR-based Place Recognition**


OverlapNetVLAD is a coase-to-fine framework for LiARD-based place recognition, which use global descriptors to propose place candidates, and use overlap prediction to determine the final match.

[[Paper]](https://arxiv.org/abs/2303.06881)

## Instructions

This code has been tested on Ubuntu 18.04 (PyTorch 1.12.1, CUDA 10.2, GeForce GTX 1080Ti).

Pretrained models in [here](https://drive.google.com/drive/folders/1LEGhH38SB9Y7ia_ovYtQ3NzqRMfwJCt1?usp=sharing).

### Requirments

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install spconv-cu102==2.1.25
pip install pyymal tqdm open3d tensorboard
```
> or directly run the install.sh scripts

We use *spconv-cu102=2.1.25*, other version may report error. 
`spconv-cu116` occors with following errors.

```
RuntimeError: Error(s) in loading state_dict for backbone:
size mismatch for dconv_down1.conv.3.weight: copying a param with shape torch.Size([11, 11, 16, 16]) from checkpoint, the shape in current model is torch.Size([16, 11, 11, 16]).
size mismatch for dconv_down1_1.conv.3.weight: copying a param with shape torch.Size([11, 11, 16, 16]) from checkpoint, the shape in current model is torch.Size([16, 11, 11, 16]).
size mismatch for dconv_down2.conv.3.weight: copying a param with shape torch.Size([7, 7, 32, 32]) from checkpoint, the shape in current model is torch.Size([32, 7, 7, 32]).
size mismatch for dconv_down2_1.conv.3.weight: copying a param with shape torch.Size([7, 7, 32, 32]) from checkpoint, the shape in current model is torch.Size([32, 7, 7, 32]).
```
And you should permute the mismatch tensor by following this [tips](https://github.com/traveller59/spconv/issues/605#issuecomment-1678641998)

## 1. Extract features

```shell
python tools/utils/gen_bev_features.py
```

## 2. Train

The training of backbone network and overlap estimation network please refs to [BEVNet](https://github.com/lilin-hitcrt/BEVNet). Here is the training of global descriptor generation network.

```shell
python train/train_netvlad.py
```

## 3. Evalute

```shell
python evaluate/evaluate.py
```

the function **evaluate_vlad** is the evaluation of the coarse seaching method using global descriptors.

## 4. practical test

### 1. train
+ it seems that there are server overfit during training.
[./docs/imgs/](20231201-215957_recall.jpg)
+ the training process is slow.

### 2. evaluate
the recall at sequence 07 is about 57.76%
|sequence|07|00|06|10|
|recall|0.57763975|0.77908218|0.98905109|0.41463415|

## Acknowledgement

Thanks to the source code of some great works such as [pointnetvlad](https://github.com/mikacuy/pointnetvlad), [PointNetVlad-Pytorch
](https://github.com/cattaneod/PointNetVlad-Pytorch), [OverlapTransformer](https://github.com/haomo-ai/OverlapTransformer) and so on.


## Citation

If you find this repo is helpful, please cite:


```
@InProceedings{Fu_2023_OverlapNetVLAD,
author = {Fu, Chencan and Li, Lin and Peng, Linpeng and Ma, Yukai and Zhao, Xiangrui and Liu, Yong},
title = {OverlapNetVLAD: A Coarse-to-Fine Framework for LiDAR-based Place Recognition},
journal={arXiv preprint arXiv:2303.06881},
year={2023}
}
```

## Todo

- [x] upload pretrained models
- [ ] add pictures
- [ ] ...