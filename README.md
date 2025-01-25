# OPS: Open Panoramic Segmentation

<p>
<a href="https://arxiv.org/pdf/2407.02685">
    <img src="https://img.shields.io/badge/PDF-arXiv-brightgreen" /></a>
<a href="https://junweizheng93.github.io/publications/OPS/OPS.html">
    <img src="https://img.shields.io/badge/Project-Homepage-red" /></a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange" /></a>
<a href="https://github.com/open-mmlab/mmsegmentation">
    <img src="https://img.shields.io/badge/Framework-mmsegmentation%201.x-yellowgreen" /></a>
<a href="https://github.com/JunweiZheng93/OPS/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
</p>


## Homepage

This project has been accepted by ECCV 2024! For more information about the project, please refer to our [project homepage](https://junweizheng93.github.io/publications/OPS/OPS.html).

## Prerequisites

### Option 1: Without Docker

Please make sure your CUDA==11.8, GCC==9, G++==9 since we need to compile some operators. You can use the following command to check your CUDA, GCC and G++ version:

```bash
nvcc --version
gcc --version
g++ --version
```

Then install all necessary packages:

```bash
conda create -n OPS python=3.9 -y
conda activate OPS
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install lit==18.1.8 numpy==1.23.1 cmake==3.30.4
pip install openmim==0.3.9
mim install mmengine==0.9.0
mim install mmcv==2.1.0
mim install mmsegmentation==1.2.2
pip install timm==0.9.8 einops==0.7.0 ftfy==6.1.1 pkbar==0.5 prettytable==3.9.0 py360convert==0.1.0 regex==2023.10.3 six==1.16.0
cd ops/models/dcnv3 && bash make.sh
```

### Option 2: With Docker

You need to download this repository first since there is a Dockerfile inside. You need to build the Docker Image with this Dockerfile.

Make sure you have installed the Nvidia Container Toolkit according to this [link](https://stackoverflow.com/questions/75118992/docker-error-response-from-daemon-could-not-select-device-driver-with-capab). Otherwise, you cannot use GPUs when running a docker container.

```bash
# build the Docker Image
docker build -t ops:ubuntu18.04 .
# run a Docker container with GPUs
docker run --gpus all -it --shm-size 10gb --name ops ops:ubuntu18.04
```

It's highly recommended that you use the above commands to create a workable environment and then follow the steps below if you are unfamiliar with Docker. Otherwise, you will have a permission problem.

```bash
# you are now in the Docker container
git clone https://github.com/JunweiZheng93/OPS.git
# compile DCNv3
cd OPS/ops/models/dcnv3 && bash make.sh && cd /OPS
```
Now you're ready to go. Please download datasets and pretrained CLIP inside the Docker container to avoid the permission problem if you're unfamiliar with Docker.

## Datasets

We train our model on COCO-Stuff164k dataset while testing on WildPASS, Matterport3D and Stanford2D3D datasets.
The dataset folder structure is as follows:

```
OPS
├── ops
├── configs
├── pretrains
│   ├── ViT-B-16.pt
├── data
│   ├── coco_stuff164k
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
│   ├── matterport3d
│   │   ├── val
│   │   │   ├── rgb
│   │   │   ├── semantic
│   ├── s2d3d
│   │   ├── area1_3_6
│   │   │   ├── rgb
│   │   │   ├── semantic
│   │   ├── area2_4
│   │   │   ├── rgb
│   │   │   ├── semantic
│   │   ├── area5
│   │   │   ├── rgb
│   │   │   ├── semantic
│   ├── WildPASS
│   │   ├── images
│   │   │   ├── val
│   │   ├── annotations
│   │   │   ├── val
├── tools
├── README.md
```

### COCO-Stuff164k

Please follow [this link](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#coco-stuff-164k) to download
and preprocess COCO-Stuff164k dataset. As for the RERP data augmentation, please use the following command:

```bash
python tools/dataset_converters/add_erp.py --shuffle
```

### WildPASS

Please follow [WildPASS official repository](https://github.com/elnino9ykl/WildPASS) to download
and preprocess WildPASS dataset.

### Matterport3D

Please follow [360BEV official repository](https://github.com/jamycheung/360BEV) to download
and preprocess Matterport3D dataset.

### Stanford2D3D

Please follow [360BEV official repository](https://github.com/jamycheung/360BEV) to download
and preprocess Stanford2D3D dataset.

## Pretrained CLIP

Please download the pretrained CLIP using [this link](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt).
Then use `tools/model_converters/clip2mmseg.py` to convert model into mmseg style:

```bash
python tools/model_converters/clip2mmseg.py path/to/the/downloaded/pretrained/model/ViT-B-16.pt pretrains/ViT-B-16.pt  # the first path is the path of the downloaded model, the second one is the converted model
```

The processed model should be placed in `pretrains` folder (see dataset folder structure).

## Checkpoints

The checkpoints can be downloaded from: <br>
[Checkpoint without RERP](https://drive.google.com/file/d/1MxM5oFZnj4OnmdeDQdXrnZ9AoFNpc2Gg/view?usp=sharing)  <br>
[Checkpoint with RERP](https://drive.google.com/file/d/1zgEXNOOHojQ7XRl-DHVGPh4lA4RtBv84/view?usp=sharing)

## Usage

### Train

Please use the following command to train the model:

```bash
bash tools/dist_train.sh <CONFIG_PATH> <GPU_NUM>
```

`<CONFIG_PATH>` should be the path of the COCO_Stuff164k config file.

### Test

Please use the following command to test the model:

```bash
bash tools/dist_test.sh <CONFIG_PATH> <CHECKPOINT_PATH> <GPU_NUM>
```

`<CONFIG_PATH>` should be the path of the WildPASS, Matterport3D or Stanford2D3D config file. `<CHECKPOINT_PATH>` should be the path of the COCO_Stuff164k checkpoint file.

## Citation
If you are interested in this work, please cite as below:

```text
@inproceedings{zheng2024open,
title={Open Panoramic Segmentation},
author={Zheng, Junwei and Liu, Ruiping and Chen, Yufan and Peng, Kunyu and Wu, Chengzhi and Yang, Kailun and Zhang, Jiaming and Stiefelhagen, Rainer},
booktitle={European Conference on Computer Vision (ECCV)},
year={2024}
}
```
