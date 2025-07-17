# USTNet
This repository contains the official implementation of USTNet: A U-Net Swin Transformer Network for Aerial Visible-to-Infrared Image Translation, along with the upgraded Aerial Visible-to-Infrared Image Dataset (AVIID). Our research presents a novel cross-modal image translation framework that leverages Swin Transformer blocks within a U-Net architecture, specifically designed for aerial remote sensing applications.
# Code of USTNet
## Requirements
- Python 3.7 or higher 
- Pytorch 1.8.0, torchvison 0.9.0 
- Tensorboard, TensorboardX, Pyyaml, Pillow, dominate, visdom, timm

# Availability of Datasets
DroneVehicle datasets can be downloaded from <[https://pan.baidu.com/s/1D4l3wXmAVSG2ywL6QLGURw?pwd=hrqf](https://pan.baidu.com/s/18CI6cbg4tRK7h7CdmapruQ?pwd=feqh)>, the code is feqh. 
## Usage
Download the USTNet code. Make the `datasets` folder and put the downloaded datasets in the `Datasets` folder. 
### Pretraining:
```
python pretrain.py --dataroot ./datasets/AVIID --name AVIID_USTNet_Pretrain  --gpu_ids 0  
```
### Training:
```
python train.py --dataroot ./datasets/AVIID --name AVIID_USTNet_Use_Pretrain  --pretrain_name AVIID_USTNet_Pretrain --which_epoch 200 --use_pretrain True --epochs_warmup 200 --epochs_anneal 200 --gpu_ids 0  
```
### No Pretraining
```
python train.py --dataroot ./datasets/DayDrone/ --name DayDrone_USTNet --epochs_warmup 100 --epochs_anneal 100 --gpu_ids 0 
```
### Testing
```
python test.py --dataroot ./datasets/AVIID --name AVIID_USTNet_Use_Pretrain --which_epoch 400 --loadSize 256 --gpu_ids 0 
```

## Evaluation
| Metric | Description | Implementation | Key Parameters |
|--------|-------------|----------------|----------------|
| **FID** <br> **KID** | Frechet Inception Distance <br> Kernel Inception Distance | [torch-fidelity](https://github.com/toshas/torch-fidelity) | `kid-subset-size=500` (for AVIID dataset) |
| **LPIPS** | Learned Perceptual Image Patch Similarity | [LPIPS PyTorch](https://github.com/richzhang/PerceptualSimilarity) | net='alex' |
| **SSIM** <br> **PSNR** <br> **RMSE** | Structural Similarity <br> Peak Signal-to-Noise Ratio <br> Root Mean Square Error | `metrics/` | - |


