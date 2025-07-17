import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from models.dynamic_parallel_window_attention import DynamicParallelWindowAttention
import functools
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class USTNetModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(
            opt.input_nc, 48,
            kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.enc_block_0 = UNetEncBlock(48, 48)
        self.enc_block_1 = UNetEncBlock(48, 96)
        self.enc_block_2 = UNetEncBlock(96, 192)
        self.enc_block_3 = UNetEncBlock(192, 384)

        self.transformer_block = DynamicParallelWindowAttention()

        self.dec_block_0 = UNetDecBlock(384, 192)
        self.dec_block_1 = UNetDecBlock(192, 96)
        self.dec_block_2 = UNetDecBlock(96, 48)
        self.dec_block_3 = UNetDecBlock(48, 48)

        self.output_layer = nn.Sequential(nn.Conv2d(
            48, opt.output_nc,
            kernel_size=1), nn.Tanh(),
        )

    def forward(self, x):
        x_input = self.input_layer(x)  # [b,48,256,256]
        x_enc_r_0, x_enc_y_0 = self.enc_block_0(x_input)  # [b,48,256,256] [b,48,128,128]
        x_enc_r_1, x_enc_y_1 = self.enc_block_1(x_enc_y_0)  # [b,96,128,128] [b,96,64,64]
        x_enc_r_2, x_enc_y_2 = self.enc_block_2(x_enc_y_1)  # [b,192,64,64] [b,192,32,32]
        x_enc_r_3, x_enc_y_3 = self.enc_block_3(x_enc_y_2)  # [b,384,32,32] [b,384,16,16]

        x_inner = self.transformer_block(x_enc_y_3)

        x_dec_y_0 = self.dec_block_0(x_inner, x_enc_r_3)  # [b,192,32,32]
        x_dec_y_1 = self.dec_block_1(x_dec_y_0, x_enc_r_2)  # [b,96,64,64]
        x_dec_y_2 = self.dec_block_2(x_dec_y_1, x_enc_r_1)  # [b,48,128,128]
        x_dec_y_3 = self.dec_block_3(x_dec_y_2, x_enc_r_0)  # [b,48,256,256]

        x_output = self.output_layer(x_dec_y_3)

        return x_output


class UnetBasicBlock(nn.Module):

    def __init__(self, in_feature, out_feature, mid_feature=None):
        super().__init__()
        if mid_feature is None:
            mid_feature = out_feature
        self.block = nn.Sequential(
            nn.InstanceNorm2d(in_feature),
            nn.Conv2d(in_feature, mid_feature, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(mid_feature),
            nn.Conv2d(mid_feature, out_feature, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)


class UNetEncBlock(nn.Module):

    def __init__(
            self, in_feature, out_feature):
        super().__init__()
        self.block = UnetBasicBlock(in_feature, out_feature, mid_feature=None)

        self.downsample = nn.Conv2d(out_feature, out_feature, kernel_size=2, stride=2)

    def forward(self, x):
        r = self.block(x)
        y = self.downsample(r)
        return r, y


class UNetDecBlock(nn.Module):

    def __init__(
            self, in_feature, out_feature):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_feature, in_feature, kernel_size=3, padding=1),
        )
        self.block = UnetBasicBlock(in_feature * 2, out_feature, mid_feature=in_feature)

    def forward(self, x, r):
        x = self.upsample(x)
        y = torch.cat([x, r], dim=1)
        y = self.block(y)
        return y


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, opt, ndf=64, n_layers=3, max_mult=8):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        norm_layer = nn.BatchNorm2d

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(opt.input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, max_mult)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, max_mult)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


def linear_scheduler(optimizer, epochs_warmup, epochs_anneal):
    def lambda_rule(epoch, epochs_warmup, epochs_anneal):
        if epoch < epochs_warmup:
            return 1.0

        return 1.0 - (epoch - epochs_warmup) / (epochs_anneal + 1)

    lr_fn = lambda epoch: lambda_rule(epoch, epochs_warmup, epochs_anneal)

    return lr_scheduler.LambdaLR(optimizer, lr_fn)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        return linear_scheduler(optimizer, opt.epochs_warmup, opt.epochs_anneal)
