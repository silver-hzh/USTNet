import numpy as np
import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from models.base_model import BaseModel
from models.networks import *
from collections import OrderedDict
import util.util as util
from models.losses import *
from torch.optim import lr_scheduler
from models.wavelet_transform import WaveletTransform2D
import torch.nn.functional as F


def calc_tokenized_size(image_shape, token_size):
    # image_shape : (C, H, W)
    # token_size  : (H_t, W_t)
    if image_shape[1] % token_size[0] != 0:
        raise ValueError(
            "Token width %d does not divide image width %d" % (
                token_size[0], image_shape[1]
            )
        )

    if image_shape[2] % token_size[1] != 0:
        raise ValueError(
            "Token height %d does not divide image height %d" % (
                token_size[1], image_shape[2]
            )
        )

    # result : (N_h, N_w)
    return (image_shape[1] // token_size[0], image_shape[2] // token_size[1])


class ImagePatchRandomMasking(nn.Module):

    def __init__(self, patch_size, fraction=0.4):
        super().__init__()

        self._patch_size = patch_size
        self._fraction = fraction

    def forward(self, image):
        # image : (N, C, H, W)
        N_h, N_w = calc_tokenized_size(image.shape[1:], (self._patch_size, self._patch_size))

        # mask : (N, 1, N_h, N_w)
        mask = (torch.rand((image.shape[0], 1, N_h, N_w)) > self._fraction)

        # mask : (N, 1, N_h, N_w)
        #     -> (N, 1,   H,   W)
        mask = mask.repeat_interleave(self._patch_size, dim=2)
        mask = mask.repeat_interleave(self._patch_size, dim=3)

        return mask.to(image.device) * image, mask


class USTNetPT(BaseModel):
    def name(self):
        return 'USTNetPT'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.use_wavelet = opt.use_wavelet

        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # generator

        self.netG_AB = USTNetModel(opt).cuda(opt.gpu_ids[0])

        self.masking = ImagePatchRandomMasking(patch_size=opt.patch_size, fraction=opt.fraction)
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionRes = torch.nn.MSELoss()

        self.model_names = ['G_AB']
        self.wavelet_transform = WaveletTransform2D(opt.input_nc, opt.gpu_ids[0])


        if self.isTrain:
            self.init_weights()

            self.optimizer_G = torch.optim.AdamW(itertools.chain(self.netG_AB.parameters()),
                                                 lr=opt.batchSize * 5e-3 / 512,
                                                 betas=(0.9, 0.99), weight_decay=0.05)
            self.scheduler_G = self.get_scheduler(self.optimizer_G, opt)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):

        self.real_A = self.input_A
        self.real_B = self.input_B

        self.mask_A, self.mask_for_A = self.masking(self.real_A)
        self.mask_B, self.mask_for_B = self.masking(self.real_B)

        self.rec_A = self.netG_AB(self.mask_A)
        self.rec_B = self.netG_AB(self.mask_B)

        if self.use_wavelet:
            self.rec_A_LL, self.rec_A_LH, self.rec_A_HL, self.rec_A_HH = self.wavelet_transform(
                self.rec_A)
            self.real_A_LL, self.real_A_LH, self.real_A_HL, self.real_A_HH = self.wavelet_transform(
                self.real_A)

            self.rec_B_LL, self.rec_B_LH, self.rec_B_HL, self.rec_B_HH = self.wavelet_transform(
                self.rec_B)
            self.real_B_LL, self.real_B_LH, self.real_B_HL, self.real_B_HH = self.wavelet_transform(
                self.real_B)

    def backward_G(self):


        loss_A = self.criterionRes(self.rec_A, self.real_A)
        loss_B = self.criterionRes(self.rec_B, self.real_B)

        if self.use_wavelet:
            loss_A_lf = self.criterionRes(self.rec_A_LL, self.real_A_LL) * self.opt.lambda_wavelet
            loss_A_hf = self.criterionRes(self.rec_A_LH + self.rec_A_HL + self.rec_A_HH,
                                          self.real_A_LH + self.real_A_HL + self.real_A_HH) * self.opt.lambda_wavelet

            loss_B_lf = self.criterionRes(self.rec_B_LL, self.real_B_LL) * self.opt.lambda_wavelet
            loss_B_hf = self.criterionRes(self.rec_B_LH + self.rec_B_HL + self.rec_B_HH,
                                          self.real_B_LH + self.real_B_HL + self.real_B_HH) * self.opt.lambda_wavelet

            loss = loss_A + loss_B + loss_A_hf + loss_B_hf + loss_A_lf + loss_B_lf
        else:
            loss = loss_A + loss_B

        loss.backward()

        self.loss_A = loss_A.item()
        self.loss_B = loss_B.item()
        if self.use_wavelet:
            self.loss_A_lf = loss_A_lf.item()
            self.loss_B_lf = loss_B_lf.item()
            self.loss_A_hf = loss_A_hf.item()
            self.loss_B_hf = loss_B_hf.item()

    def get_current_errors(self):
        if self.use_wavelet:
            ret_errors = OrderedDict(
                [('loss_A', self.loss_A), ('loss_B', self.loss_B), ('loss_A_lf', self.loss_A_lf),
                 ('loss_B_lf', self.loss_B_lf), ('loss_A_hf', self.loss_A_hf), ('loss_B_hf', self.loss_B_hf)])
        else:
            ret_errors = OrderedDict(
                [('loss_A', self.loss_A), ('loss_B', self.loss_B)])

        return ret_errors

    def optimize_parameters(self):
        self.forward()
        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def save(self, label):
        self.save_network(self.netG_AB, 'G_AB', label, self.opt.gpu_ids)

    def get_image_paths(self):
        return self.image_paths

    def test(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.mask_A = self.masking(self.real_A)
        self.mask_B = self.masking(self.real_B)
        self.rec_A = self.netG_AB(self.mask_A)
        self.rec_B = self.netG_AB(self.mask_B)
        # print(self.rec_B.size())

    def get_L1_error(self):
        loss = self.criterionRes(self.rec_A, self.real_A).item()
        loss += self.criterionRes(self.rec_B, self.rec_B).item()
        return loss

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.detach().clone())
        real_B = util.tensor2im(self.real_B.detach().clone())

        mask_A = util.tensor2im(self.mask_A.detach().clone())
        mask_B = util.tensor2im(self.mask_B.detach().clone())

        rec_A = util.tensor2im(self.rec_A.detach().clone())
        rec_B = util.tensor2im(self.rec_B.detach().clone())

        if self.use_wavelet:
            real_A_lf = util.tensor2im(self.real_A_LL.detach().clone())
            real_A_hf = util.tensor2im((self.real_A_LH + self.real_A_HL + self.real_A_HH).detach().clone())
            real_B_lf = util.tensor2im(self.real_B_LL.detach().clone())
            real_B_hf = util.tensor2im((self.real_B_LH + self.real_B_HL + self.real_B_HH).detach().clone())

            rec_A_lf = util.tensor2im(self.rec_A_LL.detach().clone())
            rec_A_hf = util.tensor2im((self.rec_A_LH + self.rec_A_HL + self.rec_A_HH).detach().clone())
            rec_B_lf = util.tensor2im(self.rec_B_LL.detach().clone())
            rec_B_hf = util.tensor2im((self.rec_B_LH + self.rec_B_HL + self.rec_B_HH).detach().clone())
            ret_visuals = OrderedDict(
                [('real_A', real_A), ('real_B', real_B), ('mask_A', mask_A), ('mask_B', mask_B), ('rec_A', rec_A),
                 ('rec_B', rec_B), ('real_A_lf', real_A_lf), ('real_A_hf', real_A_hf), ('real_B_lf', real_B_lf),
                 ('real_B_hf', real_B_hf), ('rec_A_lf', rec_A_lf), ('rec_A_hf', rec_A_hf), ('rec_B_lf', rec_B_lf),
                 ('rec_B_hf', rec_B_hf)])
        else:
            ret_visuals = OrderedDict(
                [('real_A', real_A), ('real_B', real_B), ('mask_A', mask_A), ('mask_B', mask_B), ('rec_A', rec_A),
                 ('rec_B', rec_B)])
        return ret_visuals

    def update_learning_rate(self):
        self.scheduler_G.step()
        lr = self.optimizer_G.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def init_weights(self):
        self.netG_AB.apply(self._init_weights)

    def get_scheduler(self, optimizer, opt):
        if opt.lr_policy == 'CosineAnnealingWarmRestarts':
            return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1,
                                                            eta_min=opt.batchSize * 5e-8 / 512)
