import torch
import torch.nn as nn
import numpy as np


class WaveletTransform2D(nn.Module):
    def __init__(self, in_channels, gpu_ids):
        super().__init__()
        self.in_channels = in_channels
        self.gpu_ids = gpu_ids

        # Haar wavelet filters
        self.harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
        self.harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        self.harr_wav_H[0, 0] = -1 * self.harr_wav_H[0, 0]

        self.harr_wav_LL = np.transpose(self.harr_wav_L) * self.harr_wav_L
        self.harr_wav_LH = np.transpose(self.harr_wav_L) * self.harr_wav_H
        self.harr_wav_HL = np.transpose(self.harr_wav_H) * self.harr_wav_L
        self.harr_wav_HH = np.transpose(self.harr_wav_H) * self.harr_wav_H

        # Convert filters to PyTorch tensors and move to device
        self.filter_LL = torch.from_numpy(self.harr_wav_LL).unsqueeze(0).cuda(self.gpu_ids)
        self.filter_LH = torch.from_numpy(self.harr_wav_LH).unsqueeze(0).cuda(self.gpu_ids)
        self.filter_HL = torch.from_numpy(self.harr_wav_HL).unsqueeze(0).cuda(self.gpu_ids)
        self.filter_HH = torch.from_numpy(self.harr_wav_HH).unsqueeze(0).cuda(self.gpu_ids)

        # Define convolutional layers for each transform component
        self.LL = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False,
                            groups=in_channels).cuda(self.gpu_ids)
        self.LH = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False,
                            groups=in_channels).cuda(self.gpu_ids)
        self.HL = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False,
                            groups=in_channels).cuda(self.gpu_ids)
        self.HH = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False,
                            groups=in_channels).cuda(self.gpu_ids)

        # Set weights to be non-trainable and assign filter coefficients
        with torch.no_grad():
            self.LL.weight.requires_grad = False
            self.LH.weight.requires_grad = False
            self.HL.weight.requires_grad = False
            self.HH.weight.requires_grad = False

            self.LL.weight.data = self.filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
            self.LH.weight.data = self.filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
            self.HL.weight.data = self.filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
            self.HH.weight.data = self.filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    def forward(self, input):
        LL = self.LL(input)
        LH = self.LH(input)
        HL = self.HL(input)
        HH = self.HH(input)

        return LL, LH, HL, HH
