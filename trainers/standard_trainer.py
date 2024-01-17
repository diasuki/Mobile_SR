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
from utils.metrics import AlignedPSNR
from utils.post_processing_vis import generate_processed_image_channel3
import pickle
import cv2
from data.utils.postprocessing_functions import SimplePostProcess

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
        aligned: bool = False,
        alignment_net: nn.Module = None
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
        self.aligned = aligned
        self.alignment_net = alignment_net
        if self.aligned:
            self.aligned_psnr = AlignedPSNR(alignment_net=alignment_net, boundary_ignore=40)
        self.sr_residuals = None
        self.setup_meters()
    
    def setup_meters(self):
        training_loop = self._loops_satisfy(lambda c: c.training)
        non_training_loop = self._loops_satisfy(lambda c: not c.training)
        self.add_meter('learning_rate', 'lr',
                       meter_type='scaler', omit_from_results=True)
        self.add_meter('loss', 'L', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.gw_loss_weight:
            self.add_meter('loss_gw', 'Lgw', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.fft_loss_weight:
            self.add_meter('loss_fft', 'Lfft', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.lpips_loss_weight:
            self.add_meter('loss_lpips', 'Llpips', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.msssim_loss_weight:
            self.add_meter('loss_msssim', 'Lmsssim', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.msssimgw_loss_weight:
            self.add_meter('loss_msssimgw', 'Lmsssimgw', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.msssimgwx_loss_weight:
            self.add_meter('loss_msssimgwx', 'Lmsssimgwx', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.msssimgwy_loss_weight:
            self.add_meter('loss_msssimgwy', 'Lmsssimgwy', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.gmsssim_loss_weight:
            self.add_meter('loss_gmsssim', 'Lgmsssim', loop_id=training_loop, fstr_format='7.4f')
        # if self.opt.loss1_weight:
        #     self.add_meter('loss1', 'L1', loop_id=training_loop, fstr_format='7.4f')
        # if self.opt.loss2_weight:
        #     self.add_meter('loss2', 'L2', loop_id=training_loop, fstr_format='7.4f')
        # if self.opt.loss3_weight:
        #     self.add_meter('loss3', 'L3', loop_id=training_loop, fstr_format='7.4f')
        # if self.opt.loss4_weight:
        #     self.add_meter('loss4', 'L4', loop_id=training_loop, fstr_format='7.4f')
        # if self.opt.loss5_weight:
        #     self.add_meter('loss5', 'L5', loop_id=training_loop, fstr_format='7.4f')
        # if self.opt.loss6_weight:
        #     self.add_meter('loss6', 'L6', loop_id=training_loop, fstr_format='7.4f')
        # if self.opt.loss7_weight:
        #     self.add_meter('loss7', 'L7', loop_id=training_loop, fstr_format='7.4f')
        # if self.opt.loss8_weight:
        #     self.add_meter('loss8', 'L8', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.lw_loss_weight:
            self.add_meter('loss_lw', 'Llw', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.sr_residual_loss_weight:
            self.add_meter('loss_sr_residual', 'Lsr', loop_id=training_loop, fstr_format='7.4f')
        if self.opt.sr_residual_gwloss_weight:
            self.add_meter('loss_sr_residual_gw', 'Lsrgw', loop_id=training_loop, fstr_format='7.4f')
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

        if epoch_id > 0 and epoch_id % 10 == 0:
            self.sr_residuals = torch.abs(images_HR - images_restored).clone().detach()

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
        if self.opt.fft_loss_weight:
            criterion_fft = self.criterions['fft']
            loss_fft = criterion_fft(images_restored, images_HR)
            loss += loss_fft
        if self.opt.lpips_loss_weight:
            criterion_lpips = self.criterions['lpips']
            loss_lpips = criterion_lpips(images_restored, images_HR)
            loss += self.opt.lpips_loss_weight * loss_lpips
        if self.opt.msssim_loss_weight:
            criterion_msssim = self.criterions['msssim']
            loss_msssim = criterion_msssim(images_restored, images_HR)
            loss += self.opt.msssim_loss_weight * loss_msssim
        if self.opt.msssimgw_loss_weight:
            criterion_msssimgw = self.criterions['msssimgw']
            loss_msssimgw = criterion_msssimgw(images_restored, images_HR)
            loss += self.opt.msssimgw_loss_weight * loss_msssimgw
        if self.opt.msssimgwx_loss_weight:
            criterion_msssimgwx = self.criterions['msssimgwx']
            loss_msssimgwx = criterion_msssimgwx(images_restored, images_HR)
            loss += self.opt.msssimgwx_loss_weight * loss_msssimgwx
        if self.opt.msssimgwy_loss_weight:
            criterion_msssimgwy = self.criterions['msssimgwy']
            loss_msssimgwy = criterion_msssimgwy(images_restored, images_HR)
            loss += self.opt.msssimgwy_loss_weight * loss_msssimgwy
        if self.opt.gmsssim_loss_weight:
            criterion_gmsssim = self.criterions['gmsssim']
            loss_gmsssim = criterion_gmsssim(images_restored, images_HR)
            loss += self.opt.gmsssim_loss_weight * loss_gmsssim
        # if self.opt.loss1_weight:
        #     criterion_loss1 = self.criterions['loss1']
        #     loss_loss1 = criterion_loss1(images_restored, images_HR)
        #     loss += self.opt.loss1_weight * loss_loss1
        # if self.opt.loss2_weight:
        #     criterion_loss2 = self.criterions['loss2']
        #     loss_loss2 = criterion_loss2(images_restored, images_HR)
        #     loss += self.opt.loss2_weight * loss_loss2
        # if self.opt.loss3_weight:
        #     criterion_loss3 = self.criterions['loss3']
        #     loss_loss3 = criterion_loss3(images_restored, images_HR)
        #     loss += self.opt.loss3_weight * loss_loss3
        # if self.opt.loss4_weight:
        #     criterion_loss4 = self.criterions['loss4']
        #     loss_loss4 = criterion_loss4(images_restored, images_HR)
        #     loss += self.opt.loss4_weight * loss_loss4
        # if self.opt.loss5_weight:
        #     criterion_loss5 = self.criterions['loss5']
        #     loss_loss5 = criterion_loss5(images_restored, images_HR)
        #     loss += self.opt.loss5_weight * loss_loss5
        # if self.opt.loss6_weight:
        #     criterion_loss6 = self.criterions['loss6']
        #     loss_loss6 = criterion_loss6(images_restored, images_HR)
        #     loss += self.opt.loss6_weight * loss_loss6
        # if self.opt.loss7_weight:
        #     criterion_loss7 = self.criterions['loss7']
        #     loss_loss7 = criterion_loss7(images_restored, images_HR)
        #     loss += self.opt.loss7_weight * loss_loss7
        # if self.opt.loss8_weight:
        #     criterion_loss8 = self.criterions['loss8']
        #     loss_loss8 = criterion_loss8(images_restored, images_HR)
        #     loss += self.opt.loss8_weight * loss_loss8
        if self.opt.lw_loss_weight:
            criterion_lw = self.criterions['lw']
            loss_lw = criterion_lw(images_restored, images_HR)
            loss += self.opt.lw_loss_weight * loss_lw
        if self.opt.sr_residual_loss_weight:
            criterion_sr_residual = self.criterions['sr_residual']
            loss_sr_residual = criterion_sr_residual(images_restored, images_HR, self.sr_residuals)
            loss += self.opt.sr_residual_loss_weight * loss_sr_residual
        if self.opt.sr_residual_gwloss_weight:
            criterion_sr_residual_gw = self.criterions['sr_residual_gw']
            loss_sr_residual_gw = criterion_sr_residual_gw(images_restored, images_HR, self.sr_residuals)
            loss += self.opt.sr_residual_gwloss_weight * loss_sr_residual_gw

        loss /= n_accum_steps
        loss.backward()
        if self.opt.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), self.opt.grad_clip)

        self.loop_meters['loss'].update(loss)
        if self.opt.gw_loss_weight:
            self.loop_meters['loss_gw'].update(loss_gw)
        if self.opt.fft_loss_weight:
            self.loop_meters['loss_fft'].update(loss_fft)
        if self.opt.lpips_loss_weight:
            self.loop_meters['loss_lpips'].update(loss_lpips)
        if self.opt.msssim_loss_weight:
            self.loop_meters['loss_msssim'].update(loss_msssim)
        if self.opt.msssimgw_loss_weight:
            self.loop_meters['loss_msssimgw'].update(loss_msssimgw)
        if self.opt.msssimgwx_loss_weight:
            self.loop_meters['loss_msssimgwx'].update(loss_msssimgwx)
        if self.opt.msssimgwy_loss_weight:
            self.loop_meters['loss_msssimgwy'].update(loss_msssimgwy)
        if self.opt.gmsssim_loss_weight:
            self.loop_meters['loss_gmsssim'].update(loss_gmsssim)
        # if self.opt.loss1_weight:
        #     self.loop_meters['loss1'].update(loss_loss1)
        # if self.opt.loss2_weight:
        #     self.loop_meters['loss2'].update(loss_loss2)
        # if self.opt.loss3_weight:
        #     self.loop_meters['loss3'].update(loss_loss3)
        # if self.opt.loss4_weight:
        #     self.loop_meters['loss4'].update(loss_loss4)
        # if self.opt.loss5_weight:
        #     self.loop_meters['loss5'].update(loss_loss5)
        # if self.opt.loss6_weight:
        #     self.loop_meters['loss6'].update(loss_loss6)
        # if self.opt.loss7_weight:
        #     self.loop_meters['loss7'].update(loss_loss7)
        # if self.opt.loss8_weight:
        #     self.loop_meters['loss8'].update(loss_loss8)
        if self.opt.lw_loss_weight:
            self.loop_meters['loss_lw'].update(loss_lw)
        if self.opt.sr_residual_loss_weight:
            self.loop_meters['loss_sr_residual'].update(loss_sr_residual)
        if self.opt.sr_residual_gwloss_weight:
            self.loop_meters['loss_sr_residual_gw'].update(loss_sr_residual_gw)
        if self.opt.log_train_psnr:
            if self.aligned:
                psnr_tmp, ssim_tmp, lpips_tmp = self.aligned_psnr(images_restored, images_HR)
                self.loop_meters['PSNR'].update(psnr_tmp)
            else:
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
            if self.opt.image_space == 'RAW':
                pkl_paths = data_batch['pkl_path']
                for image_restored, burst_name, pkl_path in zip(images_restored, burst_names, pkl_paths):
                    with open(pkl_path, 'rb') as f:
                        meta_data = pickle.load(f)
                    image_restored_saved = generate_processed_image_channel3(image_restored.cpu(), meta_data, return_np=True, black_level_substracted=True)
                    image_restored_saved = cv2.cvtColor(image_restored_saved, cv2.COLOR_RGB2BGR)
                    filename = f'{burst_name}-{self.opt.run_name}.png'
                    path = os.path.join(self.images_dir, filename)
                    cv2.imwrite(path, image_restored_saved)
            elif self.opt.image_space == 'RGB':
                for image_restored, burst_name in zip(images_restored, burst_names):
                    img = TF.to_pil_image(image_restored)
                    filename = f'{burst_name}-{self.opt.run_name}.png'
                    path = os.path.join(self.images_dir, filename)
                    img.save(path)
            elif self.opt.image_space == 'QuadRAW':
                ###### get linear_sr ######## 
                pkl_paths = data_batch['pkl_path']
                postprocess_fn = SimplePostProcess(return_np=True)
                for image_restored, burst_name,pkl_path in zip(images_restored, burst_names,pkl_paths):
                    with open(pkl_path, 'rb') as f:
                        meta_data = pickle.load(f)
                    filename = f'{burst_name}-{self.opt.run_name}.png'
                    path = os.path.join(self.images_dir, filename)
                    sr_ = postprocess_fn.process(image_restored, meta_data)
                    cv2.imwrite(path, sr_)
            else:
                raise NotImplementedError(f"Unknown image space: {self.opt.image_space}")