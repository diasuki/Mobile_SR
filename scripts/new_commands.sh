# model : [unet_pares4_new / uformer1_tiny_leff / uformer_leff]
# train : [syn_pretrain / pretrain / finetune / scratch]
# batch image_size epoch lr

CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RGB unet_pares4_new syn_pretrain 'bs64,ps384,ep100,lr5e-4,charb' --charbonnier --lr 5e-4 --epoch 100 --image_size 384 --batch 64

CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RGB unet_pares4_new pretrain 'mse'
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RGB unet_pares4_new pretrain 'charb' --charbonnier
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RGB unet_pares4_new pretrain 'charb,gw3' --charbonnier --gw_loss_weight 3

CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RGB unet_pares4_new finetune 'PT(charb),ep100,lr1e-4,charb' --load_run_name 'RGB-unet_pares4_new-pretrain-(charb)' --charbonnier --lr 1e-4 --epoch 100
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RGB unet_pares4_new finetune 'PT(charb,gw3),ep100,lr1e-4,charb,gw3' --load_run_name 'RGB-unet_pares4_new-pretrain-(charb,gw3)' --charbonnier --gw_loss_weight 3 --lr 1e-4 --epoch 100
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RGB unet_pares4_new finetune 'PT(charb,gw3),ep100,lr1e-4,charb' --load_run_name 'RGB-unet_pares4_new-pretrain-(charb,gw3)' --charbonnier --lr 1e-4 --epoch 100
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RGB unet_pares4_new finetune 'PT(charb,gw3),ep100,lr1e-4,charb,gw1' --load_run_name 'RGB-unet_pares4_new-pretrain-(charb,gw3)' --charbonnier --gw_loss_weight 1 --lr 1e-4 --epoch 100
# CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RGB unet_pares4_new finetune 'PT(charb,gw3),ep100,lr3e-4,charb,gw3' --load_run_name 'RGB-unet_pares4_new-pretrain-(charb,gw3)' --charbonnier --gw_loss_weight 3 --lr 3e-4 --epoch 100
# CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RGB unet_pares4_new finetune 'PT(charb,gw3),ep100,lr3e-5,charb,gw3' --load_run_name 'RGB-unet_pares4_new-pretrain-(charb,gw3)' --charbonnier --gw_loss_weight 3 --lr 3e-5 --epoch 100
# CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RGB unet_pares4_new finetune 'PT(charb,gw3),ep100,lr1e-5,charb,gw3' --load_run_name 'RGB-unet_pares4_new-pretrain-(charb,gw3)' --charbonnier --gw_loss_weight 1 --lr 1e-5 --epoch 100
# CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RGB unet_pares4_new finetune 'PT(charb,gw3),ep100,lr5e-4,charb,gw3' --load_run_name 'RGB-unet_pares4_new-pretrain-(charb,gw3)' --charbonnier --gw_loss_weight 3 --lr 5e-4 --epoch 100
# CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RGB unet_pares4_new finetune 'PT(charb,gw3),ep100,lr1e-4,charb,gw3,wd0' --load_run_name 'RGB-unet_pares4_new-pretrain-(charb,gw3)' --charbonnier --gw_loss_weight 3 --lr 1e-4 --epoch 100 --weight_decay 0

CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RGB unet_pares4_new scratch 'ep200,lr5e-4,charb' --charbonnier --lr 5e-4 --epoch 200
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RGB unet_pares4_new scratch 'ep200,lr5e-4,charb' --charbonnier --lr 5e-4 --epoch 200 --run_number 1
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RGB unet_pares4_new scratch 'ep200,lr5e-4,charb' --charbonnier --lr 5e-4 --epoch 200 --run_number 2
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RGB unet_pares4_new scratch 'ep200,lr3e-4,charb' --charbonnier --lr 3e-4 --epoch 200
# CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RGB unet_pares4_new scratch 'ep100,lr5e-4,charb,gw3' --charbonnier --gw_loss_weight 3 --lr 5e-4 --epoch 100
# CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RGB unet_pares4_new scratch 'ep100,lr3e-4,charb,gw3' --charbonnier --gw_loss_weight 3 --lr 3e-4 --epoch 100
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RGB unet_pares4_new scratch 'ep200,lr5e-4,charb,gw3' --charbonnier --gw_loss_weight 3 --lr 5e-4 --epoch 200
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RGB unet_pares4_new scratch 'ep200,lr3e-4,charb,gw3' --charbonnier --gw_loss_weight 3 --lr 3e-4 --epoch 200
#####################################################################
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh eval RGB unet_pares4_new pretrain 'charb' --save_images
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh eval RGB unet_pares4_new pretrain 'charb,gw3' --save_images

CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh eval RGB unet_pares4_new finetune 'PT(charb),ep100,lr1e-4,charb' --save_images
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh eval RGB unet_pares4_new finetune 'PT(charb,gw3),ep100,lr1e-4,charb,gw3' --save_images

CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh eval RGB unet_pares4_new scratch 'ep200,lr3e-4,charb' --save_images
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh eval RGB unet_pares4_new scratch 'ep200,lr5e-4,charb,gw3' --save_images

#####################################################################
# RAW L1 train
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RAW unet_pares4_new pretrain 'charb' --charbonnier
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RAW unet_pares4_new pretrain 'charb,gw3' --charbonnier --gw_loss_weight 3

CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RAW unet_pares4_new finetune 'PT(charb),ep100,lr1e-4,charb' --load_run_name 'RAW-unet_pares4_new-pretrain-(charb)' --charbonnier --lr 1e-4 --epoch 100
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RAW unet_pares4_new finetune 'PT(charb,gw3),ep100,lr1e-4,charb,gw3' --load_run_name 'RAW-unet_pares4_new-pretrain-(charb,gw3)' --charbonnier --gw_loss_weight 3 --lr 1e-4 --epoch 100
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RAW unet_pares4_new finetune 'PT(charb,gw3),ep100,lr3e-4,charb,gw3' --load_run_name 'RAW-unet_pares4_new-pretrain-(charb,gw3)' --charbonnier --gw_loss_weight 3 --lr 3e-4 --epoch 100

CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RAW unet_pares4_new scratch 'ep200,lr5e-4,charb' --charbonnier --lr 5e-4 --epoch 200
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RAW unet_pares4_new scratch 'ep200,lr5e-4,charb,gw3' --charbonnier --gw_loss_weight 3 --lr 5e-4 --epoch 200
#####################################################################
# RAW L1+cobi train
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RAW unet_pares4_new pretrain 'cobi' --cobi
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RAW unet_pares4_new pretrain 'cobi,gw3' --cobi --gw_loss_weight 3
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RAW unet_pares4_new pretrain 'lr1e-3,cobi,gw3' --cobi --gw_loss_weight 3 --lr 1e-3
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RAW unet_pares4_new pretrain 'lr2e-4,cobi,gw3' --cobi --gw_loss_weight 3 --lr 2e-4

CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RAW unet_pares4_new finetune 'PT(cobi),ep100,lr1e-4,cobi' --load_run_name 'RAW-unet_pares4_new-pretrain-(cobi)' --cobi --lr 1e-4 --epoch 100
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RAW unet_pares4_new finetune 'PT(cobi,gw3),ep100,lr1e-4,cobi,gw3' --load_run_name 'RAW-unet_pares4_new-pretrain-(cobi,gw3)' --cobi --gw_loss_weight 3 --lr 1e-4 --epoch 100

CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train RAW unet_pares4_new scratch 'ep200,lr5e-4,cobi' --cobi --lr 5e-4 --epoch 200
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train RAW unet_pares4_new scratch 'ep200,lr5e-4,cobi,gw3' --cobi --gw_loss_weight 3 --lr 5e-4 --epoch 200
#####################################################################
# RAW L1+cobi eval
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh eval RAW unet_pares4_new pretrain 'cobi'
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh eval RAW unet_pares4_new pretrain 'cobi,gw3'

CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh eval RAW unet_pares4_new finetune 'PT(cobi),ep100,lr1e-4,cobi' 
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh eval RAW unet_pares4_new finetune 'PT(cobi,gw3),ep100,lr1e-4,cobi,gw3' 

CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh eval RAW unet_pares4_new scratch 'ep200,lr5e-4,cobi' 
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh eval RAW unet_pares4_new scratch 'ep200,lr5e-4,cobi,gw3' 
#####################################################################





#####################################################################
# old script
#####################################################################
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train unet_res2 pretrain 'ep100,bs64' --epoch 100 --batch 64
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train unet_pares4_new pretrain 'ep100,bs32' --epoch 100 --batch 32

CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train unet_pares4_new pretrain 'ep100,bs32,lr1e-4,warmup3' --epoch 100 --batch 32 --lr 1e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train unet_pares4_new pretrain 'ep100,bs32,lr5e-4,warmup3' --epoch 100 --batch 32 --lr 5e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03
# CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train unet_pares4_new pretrain 'ep100,bs32,lr1e-3,warmup3' --epoch 100 --batch 32 --lr 1e-3 --lr_schedule 1cycle --onecycle_pct_start 0.03
# CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train unet_pares4_new pretrain 'ep100,bs32,lr5e-4,warmup3,wd1e-3' --epoch 100 --batch 32 --lr 5e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --weight_decay 1e-3
# CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train unet_pares4_new_bn pretrain 'ep100,bs32,lr5e-4,warmup3' --epoch 100 --batch 32 --lr 5e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run.sh train unet_pares4_new pretrain 'ep200,bs32,lr5e-4,warmup3' --epoch 200 --batch 32 --lr 5e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run.sh train unet_pares4_new pretrain 'ep200,bs32,lr3e-4,warmup3' --epoch 200 --batch 32 --lr 3e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train unet_pares4_new pretrain 'ep200,bs32,lr5e-4,warmup3,gc1' --epoch 200 --batch 32 --lr 5e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 1
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train unet_pares4_new pretrain 'ep200,bs32,lr1e-3,warmup3,gc1' --epoch 200 --batch 32 --lr 1e-3 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 1

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run.sh train unet_pares4_new pretrain 'ep300,bs32,lr3e-4,warmup3,gc1' --epoch 300 --batch 32 --lr 3e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 1
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train unet_pares4_new pretrain 'ep200,bs32,lr3e-4,warmup3,gc0p5' --epoch 200 --batch 32 --lr 3e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 0.5
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train unet_pares4_new pretrain 'ep200,bs32,lr3e-4,warmup3,gc0p25' --epoch 200 --batch 32 --lr 3e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 0.25
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train unet_pares4_new pretrain 'ep200,bs32,lr3e-4,warmup3,gc3' --epoch 200 --batch 32 --lr 3e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 3
# CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train unet_pares4_new pretrain 'ep200,bs32,lr3e-4,warmup3,gc5' --epoch 200 --batch 32 --lr 3e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 5



CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train unet_pares4_new pretrain 'ep200,bs32,lr3e-4,warmup3,gc1,gw3' --epoch 200 --batch 32 --lr 3e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 1 --gw_loss_weight 3

CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train unet_pares4_new pretrain 'ep100,bs32,lr3e-4,warmup3,gc1,charb,gw3' --epoch 100 --batch 32 --lr 3e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 1 --charbonnier --gw_loss_weight 3
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train unet_pares4_new pretrain 'ep100,bs32,lr1e-4,warmup3,gc1,charb,gw3' --epoch 100 --batch 32 --lr 1e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 1 --charbonnier --gw_loss_weight 3

CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train unet_pares4_new pretrain_RAW 'ep100,bs32,lr3e-4,warmup3,gc1,charb,gw3' --epoch 100 --batch 32 --lr 3e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 1 --charbonnier --gw_loss_weight 3
CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train unet_pares4_new pretrain_RAW 'ep100,bs32,lr1e-4,warmup3,gc1,charb,gw3' --epoch 100 --batch 32 --lr 1e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 1 --charbonnier --gw_loss_weight 3
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train unet_pares4_new pretrain_RAW 'ep100,bs32,lr5e-4,warmup3,gc1,charb,gw3' --epoch 100 --batch 32 --lr 5e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 1 --charbonnier --gw_loss_weight 3


CUDA_VISIBLE_DEVICES='0,1,2,3' bash run.sh train unet_pares4_new pretrain 'ep200,bs32,lr3e-4,warmup3,gc1,charb,gw3' --epoch 200 --batch 32 --lr 3e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 1 --charbonnier --gw_loss_weight 3
CUDA_VISIBLE_DEVICES='4,5,6,7' bash run.sh train unet_pares4_new pretrain 'ep200,bs32,lr5e-4,warmup3,gc1,charb,gw3' --epoch 200 --batch 32 --lr 5e-4 --lr_schedule 1cycle --onecycle_pct_start 0.03 --grad_clip 1 --charbonnier --gw_loss_weight 3