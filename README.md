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