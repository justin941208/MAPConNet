# MAPConNet: Self-supervised 3D Pose Transfer with Mesh and Point Contrastive Learning

This repository is the official implementation of our paper:

**MAPConNet: Self-supervised 3D Pose Transfer with Mesh and Point Contrastive Learning.** [[arXiv](https://arxiv.org/pdf/2304.13819.pdf)]

Jiaze Sun<sup>1</sup>, Zhixiang Chen<sup>2</sup>, Tae-Kyun Kim<sup>1,3</sup>

<sup>1</sup>Imperial College London, <sup>2</sup>University of Sheffield, <sup>3</sup>Korea Advanced Institute of Science and Technology

*Abstract: 3D pose transfer is a challenging generation task that aims to transfer the pose of a source geometry onto a target geometry with the target identity preserved. Many prior methods require keypoint annotations to find correspondence between the source and target. Current pose transfer methods allow end-to-end correspondence learning but require the desired final output as ground truth for supervision. Unsupervised methods have been proposed for graph convolutional models but they require ground truth correspondence between the source and target inputs. We present a novel self-supervised framework for 3D pose transfer which can be trained in unsupervised, semi-supervised, or fully supervised settings without any correspondence labels. We introduce two contrastive learning constraints in the latent space: a mesh-level loss for disentangling global patterns including pose and identity, and a point-level loss for discriminating local semantics. We demonstrate quantitatively and qualitatively that our method achieves state-of-the-art results in supervised 3D pose transfer, with comparable results in unsupervised and semi-supervised settings. Our method is also generalisable to unseen human and animal data with complex topologies.*

![Kiku](/figs/teaser.svg)

## Setting up the environment
- Install Anaconda with Python 3.6, then install the following dependencies:
```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
conda install -c conda-forge pymesh2
conda install -c conda-forge tqdm
```

Navigate to the `MAPConNet` root directory:
```bash
cd MAPConNet
```

Our implementation is based on [3D-CoreNet](https://github.com/chaoyuesong/3d-corenet). We also include code from [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) under `models/networks/sync_batchnorm`.

## Dataset preparation

- Our human data is the same as [NPT](https://github.com/jiashunwang/Neural-Pose-Transfer) and [3D-CoreNet](https://github.com/chaoyuesong/3d-corenet), which is generated using [SMPL](https://smpl.is.tue.mpg.de/). It can be downloaded [here](https://drive.google.com/drive/folders/11LbPXbDg4F_pSIr0sMHzWI08FOC8XvSY).

- Our animal data is the same as [3D-CoreNet](https://github.com/chaoyuesong/3d-corenet), which is generated using [SMAL](https://smal.is.tue.mpg.de/). It can be downloaded [here](https://drive.google.com/drive/folders/1uP6H0j7mUJ6utgvXxpT-2rn4EYhJ3el5?usp=sharing).

- The downloaded files should be decompressed into folders `npt-data` and `smal-data` for humans and animals respectively. These should be placed under `--dataroot`, which by default is `../data`.


## Training

There are two dataset modes, `human` and `animal`, which correspond to SMPL and SMAL data respectively. This is specified using the `--dataset_mode` option.

#### 1) Supervised learning
To train a model from scratch in a fully supervised manner, run the following command:
```bash
python train.py --dataset_mode [dataset_mode] --dataroot [parent directory of npt-data] --exp_name [name of experiment] --gpu_ids 0,1
```
#### 2) Unsupervised learning
To train a model from scratch in a fully unsupervised manner, run the following command:
```bash
python train.py --dataset_mode [dataset_mode] --dataroot [parent directory of npt-data] --exp_name [name of experiment] --gpu_ids 0,1 --percentage 0 --use_unlabelled
```
#### 3) Semi-supervised learning
To train a model from scratch in a semi-unsupervised manner where the labelled set contains 50% of all available identities and poses, run the following command:
```bash
python train.py --dataset_mode [dataset_mode] --dataroot [parent directory of npt-data] --exp_name [name of experiment] --gpu_ids 0,1 --percentage 50 --use_unlabelled
```

The checkpoints during training will be saved to `output/[dataset_mode]/[exp_name]/checkpoints/`.

Additional training options with descriptions can be found or added in the files `./options/base_options.py` and `./options/train_options`. For instance, to resume training from the latest checkpoint, append `--continue_train` to the training command.

## Testing
To test a newly trained model, run the following command:
```bash
python test.py --dataset_mode [dataset_mode] --dataroot [parent directory of npt-data] --exp_name [name of experiment] --test_epoch [the epoch to load] --metric [the metric to use] --save_output
```
The quantitative results will be saved to `.output/[dataset_mode]/[exp_name]/[epoch]/`, and the final and warped outputs are saved to `.output/[dataset_mode]/[exp_name]/[epoch]/outputs/`.

There are two options for `--metric`: `PMD` and `CD`, which are Pointwise Mesh Distance and Chamfer Distance respectively. Additional test options with descriptions can be found or added in the file `./options/test_options.py`.


## Pretrained model checkpoints
We provide pretrained checkpoints for models (D) and (N) in Table 1 of our [paper](https://arxiv.org/pdf/2304.13819.pdf), which are the best performing models on SMPL and SMAL respectively. The checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1BNUcmu35PfYx3RrmxjGejc6f8qyUiwgk?usp=share_link). The downloaded `output` folder should be put under the `MAPConNet` directory.

- To load the human model checkpoints during testing, set `--dataset_mode` to `human` and `exp_name` to `SMPL`.
- To load the animal model checkpoints during testing, set `--dataset_mode` to `animal` and `exp_name` to `SMAL`.
