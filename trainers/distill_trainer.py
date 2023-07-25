from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from trainers.base import BaseTrainer, LoopConfig, OptimizerWithSchedule
from utils.experiman import ExperiMan
from utils.metrics import batch_PSNR as psnr
from utils.metrics import mySSIM as ssim
from utils.metrics import myLPIPS as lpips

from trainers import StandardTrainer

class DistillTrainer(StandardTrainer):

    def __init__(self, *args, teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert teacher is not None
        self.teacher = teacher
    
    def do_step_train(self, epoch_id, data_batch, config, n_accum_steps):
        model = self.models['model']
        # images_LR, images_HR = data_batch
        images_LR = data_batch['LR']
        images_HR = data_batch['HR']
        if 'base frame' in data_batch:
            base_frame = data_batch['base frame']
            # print('base')
        else:
            base_frame = None

        images_restored = model(images_LR, base_frame)

        criterion = self.criterions['reconstruction']
        loss = criterion(images_restored, images_HR)
        if self.opt.gw_loss_weight:
            criterion_gw = self.criterions['gw']
            loss_gw = criterion_gw(images_restored, images_HR)
            loss += self.opt.gw_loss_weight * loss_gw
        if self.opt.lapgw_loss_weight:
            criterion_lapgw = self.criterions['lapgw']
            loss_lapgw = criterion_lapgw(images_restored, images_HR)
            loss += self.opt.lapgw_loss_weight * loss_lapgw

        loss /= n_accum_steps
        loss.backward()
        if self.opt.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), self.opt.grad_clip)

        self.loop_meters['loss'].update(loss)
        if self.opt.gw_loss_weight:
            self.loop_meters['loss_gw'].update(loss_gw)
        if self.opt.log_train_psnr:
            self.loop_meters['PSNR'].update(psnr(images_restored, images_HR))