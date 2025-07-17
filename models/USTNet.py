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


class USTNet(BaseModel):
    def name(self):
        return 'USTNet'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.use_pretrain = opt.use_pretrain

        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # generator

        self.netG_AB = USTNetModel(opt).cuda(opt.gpu_ids[0])
        self.netD_B = NLayerDiscriminator(opt).cuda(opt.gpu_ids[0])
        self.netD_gc_B = NLayerDiscriminator(opt).cuda(opt.gpu_ids[0])

        self.init_weights()

        if not self.isTrain:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_AB, 'G_AB', which_epoch, opt.gpu_ids)
            self.load_network(self.netD_B, 'D_B', which_epoch, opt.gpu_ids)
            self.load_network(self.netD_gc_B, 'D_gc_B', which_epoch, opt.gpu_ids)
        if self.use_pretrain:
            which_epoch = opt.which_epoch
            print('load model from:{}'.format(self.save_pretrain_dir))
            self.load_pretrain_network(self.netG_AB, 'G_AB', which_epoch, opt.gpu_ids)

        if self.isTrain:
            self.criterionGAN = GANLoss(gan_mode=opt.gan_mode).cuda(opt.gpu_ids[0])
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionGc = torch.nn.L1Loss()

            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_gc_B_pool = ImagePool(opt.pool_size)

            self.model_names = ['G_AB', 'D_B', 'D_gc_B']

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters()), lr=opt.lr,
                                                betas=(0.5, 0.999))
            self.optimizer_D_B = torch.optim.Adam(
                itertools.chain(self.netD_B.parameters(), self.netD_gc_B.parameters()), lr=opt.lr, betas=(0.5, 0.999))

            # initialize schedulers
            self.scheduler_G = get_scheduler(self.optimizer_G, opt)
            self.scheduler_D_B = get_scheduler(self.optimizer_D_B, opt)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        input_A = self.input_A.clone()
        input_B = self.input_B.clone()

        self.real_A = self.input_A
        self.real_B = self.input_B

        size = self.opt.fineSize

        if self.opt.geometry == 'rot':
            self.real_gc_A = self.rot90(input_A, 0)
            self.real_gc_B = self.rot90(input_B, 0)
        elif self.opt.geometry == 'vf':
            inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.opt.gpu_ids[0])
            self.real_gc_A = torch.index_select(input_A, 2, inv_idx)
            self.real_gc_B = torch.index_select(input_B, 2, inv_idx)
        elif self.opt.geometry == 'hf':
            inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.opt.gpu_ids[0])
            self.real_gc_A = torch.index_select(input_A, 3, inv_idx)
            self.real_gc_B = torch.index_select(input_B, 3, inv_idx)
        else:
            raise ValueError("Geometry transformation function [%s] not recognized." % self.opt.geometry)

    def backward_G(self):
        fake_B = self.netG_AB.forward(self.real_A)
        pred_fake = self.netD_B.forward(fake_B)
        loss_G_AB = self.criterionGAN(pred_fake, True) * self.opt.lambda_G

        fake_gc_B = self.netG_AB.forward(self.real_gc_A)
        pred_fake = self.netD_gc_B.forward(fake_gc_B)
        loss_G_gc_AB = self.criterionGAN(pred_fake, True) * self.opt.lambda_G

        if self.opt.geometry == 'rot':
            loss_gc = self.get_gc_rot_loss(fake_B, fake_gc_B, 0)
        elif self.opt.geometry == 'vf':
            loss_gc = self.get_gc_vf_loss(fake_B, fake_gc_B)
        else:
            loss_gc = self.get_gc_hf_loss(fake_B, fake_gc_B)

        if self.opt.identity > 0:
            # G_AB should be identity if real_B is fed.
            idt_A = self.netG_AB(self.real_B)
            loss_idt = self.criterionIdt(idt_A, self.real_B) * self.opt.lambda_AB * self.opt.identity
            idt_gc_A = self.netG_AB(self.real_gc_B)
            loss_idt_gc = self.criterionIdt(idt_gc_A, self.real_gc_B) * self.opt.lambda_AB * self.opt.identity

            self.idt_A = idt_A.data
            self.idt_gc_A = idt_gc_A.data
            self.loss_idt = loss_idt.item()
            self.loss_idt_gc = loss_idt_gc.item()
        else:
            loss_idt = 0
            loss_idt_gc = 0
            self.loss_idt = 0
            self.loss_idt_gc = 0

        loss_G = loss_G_AB + loss_G_gc_AB + loss_gc + loss_idt + loss_idt_gc

        loss_G.backward()

        self.fake_B = fake_B
        self.fake_gc_B = fake_gc_B

        self.loss_G_AB = loss_G_AB.item()
        self.loss_G_gc_AB = loss_G_gc_AB.item()
        self.loss_gc = loss_gc.item()

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_real = self.criterionGAN(pred_real, True)

        with torch.no_grad():
            fake = fake.contiguous()
        pred_fake = netD(fake)
        loss_fake = self.criterionGAN(pred_fake, False)

        loss = (loss_real + loss_fake) * 0.5

        if self.opt.gradient_penalty:
            # print(True)
            loss += cal_gradient_penalty(
                netD, real, fake, real.device, constant=self.opt.constant, lambda_gp=self.opt.lambda_gp
            )[0]

        loss.backward()
        return loss

    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_gc_B = self.fake_gc_B_pool.query(self.fake_gc_B)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B)
        loss_D_gc_B = self.backward_D_basic(self.netD_gc_B, self.real_gc_B, fake_gc_B)
        self.loss_D_B = loss_D_B.item()
        self.loss_D_gc_B = loss_D_gc_B.item()

    def rot90(self, tensor, direction):
        tensor = tensor.transpose(2, 3)
        size = self.opt.fineSize
        inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.opt.gpu_ids[0])
        if direction == 0:
            tensor = torch.index_select(tensor, 3, inv_idx)
        else:
            tensor = torch.index_select(tensor, 2, inv_idx)
        return tensor

    def get_image_paths(self):
        return self.image_paths

    def get_gc_rot_loss(self, AB, AB_gc, direction):
        loss_gc = 0.0

        if direction == 0:
            AB_gt = self.rot90(AB_gc.clone().detach(), 1)
            loss_gc = self.criterionGc(AB, AB_gt)
            AB_gc_gt = self.rot90(AB.clone().detach(), 0)
            loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
        else:
            AB_gt = self.rot90(AB_gc.clone().detach(), 0)
            loss_gc = self.criterionGc(AB, AB_gt)
            AB_gc_gt = self.rot90(AB.clone().detach(), 1)
            loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc

        return loss_gc

    def get_gc_vf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = self.opt.fineSize

        inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.opt.gpu_ids[0])

        AB_gt = torch.index_select(AB_gc.clone().detach(), 2, inv_idx)
        loss_gc = self.criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 2, inv_idx)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc
        return loss_gc

    def get_gc_hf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = self.opt.fineSize

        inv_idx = torch.arange(size - 1, -1, -1).long().cuda(self.opt.gpu_ids[0])

        AB_gt = torch.index_select(AB_gc.clone().detach(), 3, inv_idx)
        loss_gc = self.criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 3, inv_idx)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc

        return loss_gc

    def get_current_errors(self):

        ret_errors = OrderedDict(
            [('D_B', self.loss_D_B), ('D_gc_B', self.loss_D_gc_B),
             ('G_AB', self.loss_G_AB), ('G_gc_AB', self.loss_G_gc_AB),
             ('Gc', self.loss_gc)])

        if self.opt.identity > 0.0:
            ret_errors['idt'] = self.loss_idt
            ret_errors['idt_gc'] = self.loss_idt_gc

        return ret_errors

    def set_requires_grad(self, models, requires_grad=False):
        # pylint: disable=no-self-use
        if not isinstance(models, list):
            models = [models, ]

        for model in models:
            for param in model.parameters():
                param.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward()
        # update G
        self.set_requires_grad([self.netD_B, self.netD_gc_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # update D

        self.set_requires_grad([self.netD_B, self.netD_gc_B], True)
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def save(self, label):
        self.save_network(self.netG_AB, 'G_AB', label, self.opt.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.opt.gpu_ids)
        self.save_network(self.netD_gc_B, 'D_gc_B', label, self.opt.gpu_ids)

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.detach().clone())
        real_B = util.tensor2im(self.real_B.detach().clone())
        fake_B = util.tensor2im(self.fake_B.detach().clone())
        ret_visuals = OrderedDict(
            [('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
        return ret_visuals

    def test(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.fake_B = self.netG_AB(self.real_A)

    def update_learning_rate(self):
        self.scheduler_G.step()
        self.scheduler_D_B.step()
        lr = self.optimizer_D_B.param_groups[0]['lr']
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
        self.netD_B.apply(self._init_weights)
        self.netD_gc_B.apply(self._init_weights)
