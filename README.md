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
