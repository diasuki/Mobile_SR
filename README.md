环境配置还有运行命令可以参考scripts目录下的文件

环境配置：scripts/create_conda_env.sh 、scripts/requirements_pip.txt

运行命令：scripts/new_commands.sh

### 目录概述

- data：数据接口
- loeses：损失函数
- models：模型
- pwcnet：光流估计预训练模型
- scripts：运行命令
- trainers：模型训练
- utils：通用功能

运行命令：
```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RGB unet_pares4_new pretrain 'charb' --charbonnier
```

合成预训练：
```bash
# pretrain
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run.sh train RGB unet_pares4_new syn_pretrain 'bs128,ps384,ep200,lr2e-4,charb' --charbonnier --lr 2e-4 --epoch 200 --image_size 384 --batch 128
# finetune
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run.sh train RGB unet_pares4_new pretrain 'synPT(bs128,ps384,ep200,lr2e-4,charb),ep200,lr2e-4,charb' --load_run_name 'RGB-unet_pares4_new-syn_pretrain-(bs128,ps384,ep200,lr2e-4,charb)' --charbonnier --lr 2e-4 --epoch 200
```

quad bayer相关命令：
```bash
# quad bayer baseline
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_raw.sh train QuadRAW quadraw_unet_pares4_new quad_pretrain 'ep200,lr2e-4,charb' --charbonnier --lr 2e-4 --epoch 200
# quad bayer baseline + gw
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_raw.sh train QuadRAW quadraw_unet_pares4_new quad_pretrain 'ep200,lr2e-4,charb,gw3' --charbonnier --gw_loss_weight 3 --lr 2e-4 --epoch 200
# quad bayer baseline + msssim
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_raw.sh train QuadRAW quadraw_unet_pares4_new quad_pretrain 'ep200,lr2e-4,charb,msssim1' --charbonnier --msssim_loss_weight 1 --lr 2e-4 --epoch 200
# quad bayer baseline + newfusion
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_raw.sh train QuadRAW quadraw_newfusion_unet_pares4_new quad_pretrain 'ep200,lr2e-4,charb' --charbonnier --lr 2e-4 --epoch 200
# quad bayer baseline + pretrain
# (pretrain)
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_raw.sh train QuadRAW quadraw_unet_pares4_new quad_syn_pretrain 'bs64,ps384,ep200,lr2e-4,charb' --charbonnier --lr 2e-4 --epoch 200 --image_size 384 --batch 64
# (finetune) + gw
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_raw.sh train QuadRAW quadraw_unet_pares4_new quad_pretrain 'syn(bs64,ps384,ep200,lr2e-4,charb),bs32,ps384,ep200,lr2e-4,charb,gw3' --charbonnier --gw_loss_weight 3 --lr 2e-4 --epoch 200 --image_size 384 --batch 32 --load_run_name 'QuadRAW-quadraw_unet_pares4_new-quad_syn_pretrain-(bs64,ps384,ep200,lr2e-4,charb)'
```

其中的一些改进：

1. GW Loss : losses/gw_loss.py 中的 GWLoss (使用方式 : 1\*charb + 3\*gw)

2. 合成预训练 : data/realbsr.py 中的 QuadDataset (重点是 data/datasets/synthetic_burst_train_set.py 中的 SyntheticBurstQuadAligned，包含quad bayer多帧数据的生成+多帧数据homography对齐)

3. 模型改进 : models/unet.py 中的 UNet_NewFusion_QuadRAW (Baseline模型是 UNet_QuadRAW)

正在尝试的改进：

1. 改进 MSSSIM Loss

2. 模型改进

3. 合成预训练数据