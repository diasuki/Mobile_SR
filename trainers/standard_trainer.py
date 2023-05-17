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


class StandardTrainer(BaseTrainer):

    def __init__(
        self,
        manager: ExperiMan,
        models: dict[str, nn.Module],
        criterions: dict[str, nn.Module],
        n_epochs: int,
        loop_configs: list[LoopConfig],
        optimizers: dict[str, OptimizerWithSchedule],
        log_period: int,
        ckpt_period: int,
        device: torch.device,
        save_init_ckpt: bool = False,
        resume_ckpt: dict = None,
        keep_eval_mode: bool = False,
    ):
        self.opt = manager.get_opt()
        super().__init__(
            manager=manager,
            models=models,
            criterions=criterions,
            n_epochs=n_epochs,
            loop_configs=loop_configs,
            optimizers=optimizers,
            log_period=log_period,
            ckpt_period=ckpt_period,
            device=device,
            save_init_ckpt=save_init_ckpt,
            resume_ckpt=resume_ckpt,
        )
        self.keep_eval_mode = keep_eval_mode
        self.setup_meters()
    
    def setup_meters(self):
        training_loop = self._loops_satisfy(lambda c: c.training)
        non_training_loop = self._loops_satisfy(lambda c: not c.training)
        self.add_meter('learning_rate', 'lr',
                       meter_type='scaler', omit_from_results=True)
        self.add_meter('loss', 'L', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.gw_loss_weight:
            self.add_meter('loss_gw', 'Lgw', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.log_train_psnr:
            self.add_meter('PSNR', fstr_format='7.4f')
        else:
            self.add_meter('PSNR', loop_id=non_training_loop, fstr_format='7.4f')
        self.add_meter('SSIM', loop_id=non_training_loop, fstr_format='6.4f')
        self.add_meter('LPIPS', loop_id=non_training_loop, fstr_format='6.4f')
        loop_for_best_meter = self._loops_satisfy(lambda c: c.for_best_meter)
        if loop_for_best_meter:
            loop_id = loop_for_best_meter[0]
            self.set_meter_for_best_checkpoint(
                loop_id=loop_id, name='PSNR', maximum=True)

    def get_data_batch(self, loop_id, phase_id):
        batch = self._next_data_batch(loop_id)
        # batch = [batch['LR'], batch['HR']]
        # return [t.to(self.device) for t in batch]
        for name in batch:
            if isinstance(batch[name], torch.Tensor):
                batch[name] = batch[name].to(self.device)
        return batch

    def get_active_optimizers(self, loop_id, phase_id):
        if self.loop_configs[loop_id].training:
            return [self.optimizers['optimizer']]
        else:
            return []

    def get_checkpoint(self, epoch_id):
        checkpoint = super().get_checkpoint(epoch_id)
        loop_for_ckpt_meter = self._loops_satisfy(lambda c: c.for_ckpt_meter)
        if loop_for_ckpt_meter:
            loop_id = loop_for_ckpt_meter[0]
            if self._should_run_loop(epoch_id, loop_id):
                meters = self.meters[loop_id]
                for meter_name in ('PSNR', 'SSIM', 'LPIPS'):
                    if meter_name in meters:
                        checkpoint[f'test_{meter_name}'] = \
                            meters[meter_name].get_value()
        return checkpoint

    def toggle_model_mode(self, epoch_id, loop_id):
        model = self.models['model']
        training = self.loop_configs[loop_id].training
        model.train(training and not self.keep_eval_mode)

    def update_meters(self):
        if self.optimizers:
            lr = self.optimizers['optimizer'].get_learning_rates()[0]
            self.loop_meters['learning_rate'].update(lr)

    def do_step(self, epoch_id, loop_id, iter_id, phase_id, data_batch):
        config = self.loop_configs[loop_id]
        if config.training:
            self.do_step_train(epoch_id, data_batch, config,
                               n_accum_steps=config.n_computation_steps[phase_id])
        else:
            self.do_step_test(data_batch, config)

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

        images_restored = model(images_LR, base_frame)
        # images_restored = torch.clamp(images_restored, 0, 1)

        n = len(images_HR)
        self.loop_meters['PSNR'].update(
            psnr(images_restored, images_HR, average=False), n)
        self.loop_meters['SSIM'].update(
            ssim(images_restored, images_HR) * n, n)
        self.loop_meters['LPIPS'].update(
            lpips(images_restored, images_HR) * n, n)

class StandardTester(StandardTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.opt.save_images:
            self.images_dir = os.path.join(self.manager.get_run_dir(), 'images')
            if self.manager.is_master():
                os.makedirs(self.images_dir)

    def do_step_test(self, data_batch, config):
        model = self.models['model']

        images_LR = data_batch['LR']
        images_HR = data_batch['HR']
        if 'base frame' in data_batch:
            base_frame = data_batch['base frame']
        else:
            base_frame = None

        images_restored = model(images_LR, base_frame)
        images_restored = torch.clamp(images_restored, 0, 1)
        n = len(images_HR)
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
