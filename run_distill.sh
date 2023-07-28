trap "exit" INT
LOG_DIR="/home/zhaoqing/mobile_sr/log/super_resolution/mobile_sr"
DATA_DIR="/home/zhaoqing/dataset/RealBSR_aligned"
# MODEL_DIR="$HOME/model"
NUM_WORKERS=8

# EXP_NAME="debug"
EXP_NAME="realbsr_aligned_distill"

mode=$1; shift

space=$1; shift

basic_args="--code_dir ./ --data_dir $DATA_DIR --log_dir $LOG_DIR --num_workers $NUM_WORKERS --exp_name $EXP_NAME"
basic_args="$basic_args --image_space $space"

model=$1; shift
# model='unet'
model_args="--arch $model"
if [[ $space = 'RAW' ]]; then
    model_args="$model_args --in_channel 4 --scale 8"
fi

methods=$1; shift

# optimizer=$1; shift
# lr=$1; shift
# schedule=$1; shift
# epoch=$1; shift
# optim_args="--optimizer $optimizer --lr $lr --lr_schedule $schedule --epoch $epoch"
optim_args=""

desc=$1; shift

do_train () {
    num_devices=$(python -c 'import torch; print(torch.cuda.device_count())')
    port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    # OMP_NUM_THREADS=2 \
    # python -m torch.distributed.launch \
    #     --use_env --nnodes=1 --nproc_per_node=$num_devices --master_port $port \
    #     ./main_distill.py  $basic_args $model_args "$@"

    # OMP_NUM_THREADS=2 \
    # torchrun --standalone --nnodes=1 --nproc_per_node=$num_devices --master_port $port \
    #     ./main_distill.py  $basic_args $model_args $optim_args "$@"

    OMP_NUM_THREADS=2 \
    torchrun --nnodes=1 --nproc_per_node=$num_devices --rdzv_endpoint="localhost:$port" \
        ./main_distill.py  $basic_args $model_args $optim_args "$@"

    # pyinstrument -r html --outfile pf.html  \
    # python \
    #     ./main_distill.py  $basic_args $model_args "$@"
}
do_test () {
    num_devices=$(python -c 'import torch; print(torch.cuda.device_count())')
    port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    OMP_NUM_THREADS=2 \
    torchrun --nnodes=1 --nproc_per_node=$num_devices --rdzv_endpoint="localhost:$port" \
        ./eval_distill.py  $basic_args $model_args "$@"

    # python ./eval_distill.py  $basic_args $model_args "$@"
}
dummy(){ unused(){ :;} }

if [[ $mode = 'train' ]]; then
    train=do_train
    test=dummy
elif [[ $mode = 'eval' ]]; then
    train=dummy
    test=do_test
# elif [[ $mode = 'train-eval' ]]; then
#     train=do_train
#     test=do_test
else
    echo "Incorrect script mode '$mode', should be: train / eval"
    exit 1
fi

for method in $methods; do

    RUN_NAME="$space-$model-$method-($desc)"

    if [[ $method = 'syn_pretrain' ]]; then
        $train --run_name $RUN_NAME \
            --dataset SyntheticBurstDF2K --image_space $space \
            "$@"

    elif [[ $method = 'syn_pretrain_small' ]]; then
        $train --run_name $RUN_NAME \
            --dataset SyntheticBurstDIV2K --image_space $space \
            "$@"

    elif [[ $method = 'pretrain' ]]; then
        $train --run_name $RUN_NAME \
            --dataset RealBSR --image_space $space \
            "$@"
    # elif [[ $method = 'pretrain_RAW' ]]; then
    #     $train --run_name $RUN_NAME \
    #         --dataset RealBSR --image_space $space \
    #         --in_channel 4 --scale 8 \
    #         "$@"
    elif [[ $method = 'finetune' ]]; then
        $train --run_name $RUN_NAME \
            --dataset RealBSR_text --image_space $space \
            "$@"
    # elif [[ $method = 'finetune_RAW' ]]; then
    #     $train --run_name $RUN_NAME \
    #         --dataset RealBSR_text --image_space $space \
    #         --in_channel 4 --scale 8 \
    #         "$@"
    elif [[ $method = 'scratch' ]]; then
        $train --run_name $RUN_NAME \
            --dataset RealBSR_text --image_space $space \
            "$@"
    fi

    $test --run_name "eval-$RUN_NAME" --load_run_name $RUN_NAME \
        --dataset RealBSR_text --image_space $space \
        "$@"

done