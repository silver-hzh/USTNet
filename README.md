# USTNet
This repository contains the official implementation of USTNet: A U-Net Swin Transformer Network for Aerial Visible-to-Infrared Image Translation, along with the upgraded Aerial Visible-to-Infrared Image Dataset (AVIID). Our research presents a novel cross-modal image translation framework that leverages Swin Transformer blocks within a U-Net architecture, specifically designed for aerial remote sensing applications.
# Code of USTNet
## Requirements
- Python 3.7 or higher 
- Pytorch 1.8.0, torchvison 0.9.0 
- Tensorboard, TensorboardX, Pyyaml, Pillow, dominate, visdom, timm
## Usage
Download the USTNet code. Make the `Datasets` folder and put the downloaded datasets in the `Datasets` folder. 
### Pretraining:
```
python pretrain.py --dataroot ../Datasets/AVIID --name AVIID_USTNet_Pretrain  --gpu_ids 0  
```
### Training:
```
python train.py --dataroot ../Datasets/AVIID --name AVIID_USTNet   --gpu_ids 0  --
```
### No Pretraining
