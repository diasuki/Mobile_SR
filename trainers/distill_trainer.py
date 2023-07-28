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

from trainers import StandardTrainer, StandardTester

class DistillTrainer(StandardTrainer):

    def __init__(self, *args, teacher=None, inter_features=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert teacher is not None
        self.teacher = teacher
        self.inter_features = inter_features
        if self.inter_features:
            self.lamda = [1e-4, 1e-3, 1e-2, 1e-1]
    
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

        student_feature, images_restored = model(images_LR, base_frame)

        # teacher model
        with torch.no_grad():
            teacher_feature, images_restored_teacher = self.teacher(images_LR)

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

        # Loss(SR_student, SR_teacher)
        # print(f"loss : {loss.item()}")
        loss1 = criterion(images_restored, images_restored_teacher)
        # print(f"loss1 : {loss1.item()}")
        loss += loss1
        # Loss(inter_features)
        if self.inter_features:
            feat_num = len(teacher_feature)
            for i in range(feat_num):
                loss2 = self.lamda[i] * criterion(student_feature[i], teacher_feature[i])
                # print(f"loss2 : {loss2.item()}")
                loss += loss2

        loss /= n_accum_steps
        loss.backward()
        if self.opt.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), self.opt.grad_clip)

        self.loop_meters['loss'].update(loss)
        if self.opt.gw_loss_weight:
            self.loop_meters['loss_gw'].update(loss_gw)
        if self.opt.log_train_psnr:
            self.loop_meters['PSNR'].update(psnr(images_restored, images_HR))

    def do_step_test(self, data_batch, config):
        model = self.models['model']

        images_LR = data_batch['LR']
        images_HR = data_batch['HR']
        if 'base frame' in data_batch:
            base_frame = data_batch['base frame']
            # print('base')
        else:
            base_frame = None

        _, images_restored = model(images_LR, base_frame)
        # images_restored = torch.clamp(images_restored, 0, 1)

        n = len(images_HR)
        if self.aligned:
            psnr_tmp, ssim_tmp, lpips_tmp = self.aligned_psnr(images_restored, images_HR)
            self.loop_meters['PSNR'].update(psnr_tmp * n, n)
            self.loop_meters['SSIM'].update(ssim_tmp * n, n)
            self.loop_meters['LPIPS'].update(lpips_tmp * n, n)
        else:
            self.loop_meters['PSNR'].update(
                psnr(images_restored, images_HR, average=False), n)
            self.loop_meters['SSIM'].update(
                ssim(images_restored, images_HR) * n, n)
            self.loop_meters['LPIPS'].update(
                lpips(images_restored, images_HR) * n, n)


class DistillTester(StandardTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def do_step_test(self, data_batch, config):
        model = self.models['model']

        images_LR = data_batch['LR']
        images_HR = data_batch['HR']
        if 'base frame' in data_batch:
            base_frame = data_batch['base frame']
        else:
            base_frame = None

        _, images_restored = model(images_LR, base_frame)
        images_restored = torch.clamp(images_restored, 0, 1)
        n = len(images_HR)
        if self.aligned:
            psnr_tmp, ssim_tmp, lpips_tmp = self.aligned_psnr(images_restored, images_HR)
            self.loop_meters['PSNR'].update(psnr_tmp * n, n)
            self.loop_meters['SSIM'].update(ssim_tmp * n, n)
            self.loop_meters['LPIPS'].update(lpips_tmp * n, n)
        else:
            self.loop_meters['PSNR'].update(
                psnr(images_restored, images_HR, average=False), n)
            self.loop_meters['SSIM'].update(
                ssim(images_restored, images_HR) * n, n)
            self.loop_meters['LPIPS'].update(
                lpips(images_restored, images_HR) * n, n)

        if self.opt.save_images:
            burst_names = data_batch['burst_name']
            for image_restored, burst_name in zip(images_restored, burst_names):
                img = TF.to_pil_image(image_restored)
                filename = f'{burst_name}-{self.opt.run_name}.png'
                path = os.path.join(self.images_dir, filename)
                img.save(path)