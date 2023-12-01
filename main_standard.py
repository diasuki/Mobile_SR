# import sys
# print(sys.argv)
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils.experiman import manager
from data import *
from models import get_model
from losses import GWLoss, CharbonnierLoss, Adaptive_GWLoss, LapGWLoss, AlignedL1, AlignedGWLoss, FFTLoss, LPIPS_loss, MSSSIMLoss, LWLoss, SR_Residual_Loss, SR_Residual_GWLoss #, Loss1, Loss2, Loss3, Loss4, Loss5, Loss6, Loss7, Loss8
from trainers import StandardTrainer, LoopConfig
from utils.misc import parse
from utils.optim import get_optim
import utils.metrics
import datetime


def add_parser_argument(parser):
    ## ======================== Data ==========================
    parser.add_argument('--dataset', type=str)
    # parser.add_argument('--val_size', default=1024, type=int)
    # parser.add_argument('--data_split_seed', default=0, type=int)
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--image_space', default='RGB', type=str)
    parser.add_argument('--image_size', default=160, type=int)
    parser.add_argument('--burst_size', default=14, type=int)
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--scale', default=4, type=int)
    # parser.add_argument('--transform', type=str, default='std')
    ## ======================= Model ==========================
    parser.add_argument('--arch', type=str)
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--load_ckpt', type=str)
    parser.add_argument('--load_run_name', type=str)
    parser.add_argument('--load_run_number', type=str)
    parser.add_argument('--load_run_ckpt_name', type=str, default='ckpt-last')
    parser.add_argument('--sync_bn', action='store_true')
    ## ===================== Training =========================
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--resume_ckpt', type=str)
    parser.add_argument('--charbonnier', action='store_true')
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--aligned_gw_loss_weight', type=float)
    parser.add_argument('--cx', action='store_true')
    parser.add_argument('--cobi', action='store_true')
    parser.add_argument('--gw_loss_weight', type=float)
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--lapgw_loss_weight', type=float)
    parser.add_argument('--fft_loss_weight', type=float)
    parser.add_argument('--lpips_loss_weight', type=float)
    parser.add_argument('--msssim_loss_weight', type=float)
    # parser.add_argument('--loss1_weight', type=float)
    # parser.add_argument('--loss2_weight', type=float)
    # parser.add_argument('--loss3_weight', type=float)
    # parser.add_argument('--loss4_weight', type=float)
    # parser.add_argument('--loss4_lambda_weight', type=float)
    # parser.add_argument('--loss5_weight', type=float)
    # parser.add_argument('--loss5_lambda_weight', type=float)
    # parser.add_argument('--loss6_weight', type=float)
    # parser.add_argument('--loss7_weight', type=float)
    # parser.add_argument('--loss7_lambda_weight', type=float)
    # parser.add_argument('--loss8_weight', type=float)
    # parser.add_argument('--loss8_lambda_weight', type=float)
    parser.add_argument('--lw_loss_weight', type=float)
    parser.add_argument('--sr_residual_loss_weight', type=float)
    parser.add_argument('--sr_residual_gwloss_weight', type=float)
    ## ==================== Optimization ======================
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--num_iters_train', type=int,
                        help="default: len(trainloader)")
    parser.add_argument('--num_iters_test', type=int,
                        help="default: len(testloader)")
    parser.add_argument('--num_iters_trainset_test', type=int,
                        help="default: len(raw_trainloader)")
    parser.add_argument('--accum_steps', default=1, type=int)
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--lr_schedule', default='1cycle', type=str)
    parser.add_argument('--adam_beta', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--multistep_milestones', type=int, nargs='+')
    parser.add_argument('--cyclic_step', type=float)
    parser.add_argument('--onecycle_pct_start', default=0.03, type=float)
    parser.add_argument('--grad_clip', default=1, type=float)
    ## ====================== Logging =========================
    parser.add_argument('--log_train_psnr', action='store_true')
    parser.add_argument('--log_period', default=5, type=int, metavar='LP',
                        help='log every LP iterations')
    parser.add_argument('--ckpt_period', type=int, metavar='CP',
                        help='make checkpoints every CP epochs')
    parser.add_argument('--test_period', default=1, type=int, metavar='TP',
                        help='test every TP epochs')
    # parser.add_argument('--trainset_test_period', type=int, metavar='TP',
    #                     help='test on training set every TP epochs')
    parser.add_argument('--comment', default='', type=str)
    ## ==================== Experimental ======================


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(local_rank)
    torch.cuda.set_device(device)

    # Parse arguments and setup ExperiMan
    parser = manager.get_basic_arg_parser()
    add_parser_argument(parser)
    opt = parser.parse_args()
    if opt.resume_ckpt or opt.auto_resume:
        opt.option_for_existing_dir = 'k'
    manager.setup(opt, rank=rank, world_size=world_size,
                  third_party_tools=('tensorboard',))
    if world_size > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=10800))
        if rank == 0:
            t = torch.tensor([opt.run_number + .1], device=device)
        else:
            t = torch.empty(1, device=device)
        dist.broadcast(t, src=0)
        opt.run_number = int(t.item())
        manager.set_run_dir(manager.get_run_dir(opt.run_name, opt.run_number))
    logger = manager.get_logger()
    logger.info(f'==> Number of devices: {world_size}')

    # Data
    logger.info('==> Preparing data')
    dataset = get_dataset(
        opt.dataset, opt.data_dir, space=opt.image_space,
        size=opt.image_size, burst_size=opt.burst_size)
    assert opt.batch % world_size == 0
    batch = opt.batch // world_size
    data_kwargs = dict(
        batch_size=batch, num_workers=opt.num_workers,
        # train_split=opt.train_split, val_size=opt.val_size,
        # split_seed=opt.data_split_seed,
        world_size=world_size, rank=rank)
    # if opt.val_size > 0:
    #     trainloader, raw_trainloader, valloader, testloader = \
    #         dataset.get_loader(**data_kwargs)
    # else:
    #     trainloader, raw_trainloader, testloader = \
    #         dataset.get_loader(**data_kwargs)
    #     valloader = []
    trainloader = dataset.get_loader(split='train', **data_kwargs)
    testloader = dataset.get_loader(split='val', **data_kwargs)
    num_iters_train = parse(opt.num_iters_train, len(trainloader) // opt.accum_steps)
    # num_iters_val = len(valloader)
    # num_iters_trainset_test = parse(opt.num_iters_trainset_test, len(raw_trainloader))
    num_iters_test = parse(opt.num_iters_test, len(testloader))

    # Model
    logger.info('==> Building models')
    model = get_model(
        arch=opt.arch, dim=opt.dim, burst_size=opt.burst_size,
        in_channel=opt.in_channel, scale=opt.scale,
    ).to(device)
    model = torch.compile(model)
    # model = torch.compile(model, mode='reduce-overhead')
    if world_size > 1:
        if opt.sync_bn:
            logger.info('==> Using SyncBN')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[local_rank])
    models = {'model': model}

    # Criterions
    criterions = {}
    if opt.aligned:
        from pwcnet.pwcnet import PWCNet
        alignment_net = PWCNet(load_pretrained=True,
                           weights_path='./pwcnet/pwcnet-network-default.pth')
        alignment_net = alignment_net.to('cuda')
        for param in alignment_net.parameters():
            param.requires_grad = False
        opt.alignment_net = alignment_net
        if opt.aligned_gw_loss_weight:
            criterions['reconstruction'] = AlignedGWLoss(alignment_net=alignment_net, aligned_gw_loss_weight=opt.aligned_gw_loss_weight, boundary_ignore=40)
        else:
            criterions['reconstruction'] = AlignedL1(alignment_net=alignment_net, boundary_ignore=40)
    # elif opt.cx:
    #     criterions['reconstruction'] = L1_with_CX(boundary_ignore=40)
    # elif opt.cobi:
    #     criterions['reconstruction'] = L1_with_CoBi(boundary_ignore=40)
    elif opt.charbonnier:
        criterions['reconstruction'] = CharbonnierLoss()
    else:
        criterions['reconstruction'] = nn.MSELoss()
    if opt.gw_loss_weight:
        if opt.adaptive:
            criterions['gw'] = Adaptive_GWLoss()
        else:
            criterions['gw'] = GWLoss()
    if opt.lapgw_loss_weight:
        criterions['lapgw'] = LapGWLoss()
    if opt.fft_loss_weight:
        criterions['fft'] = FFTLoss(loss_weight=opt.fft_loss_weight)
    if opt.lpips_loss_weight:
        criterions['lpips'] = LPIPS_loss()
    if opt.msssim_loss_weight:
        criterions['msssim'] = MSSSIMLoss()
    if opt.sr_residual_loss_weight:
        criterions['sr_residual'] = SR_Residual_Loss()
    if opt.sr_residual_gwloss_weight:
        criterions['sr_residual_gw'] = SR_Residual_GWLoss()
    # if opt.loss1_weight:
    #     criterions['loss1'] = Loss1()
    # if opt.loss2_weight:
    #     criterions['loss2'] = Loss2()
    # if opt.loss3_weight:
    #     criterions['loss3'] = Loss3()
    # if opt.loss4_weight:
    #     criterions['loss4'] = Loss4(lambda_weight=opt.loss4_lambda_weight)
    # if opt.loss5_weight:
    #     criterions['loss5'] = Loss5(lambda_weight=opt.loss5_lambda_weight)
    # if opt.loss6_weight:
    #     criterions['loss6'] = Loss6()
    # if opt.loss7_weight:
    #     criterions['loss7'] = Loss7(lambda_weight=opt.loss7_lambda_weight)
    # if opt.loss8_weight:
    #     criterions['loss8'] = Loss8(lambda_weight=opt.loss8_lambda_weight)
    if opt.lw_loss_weight:
        criterions['lw'] = LWLoss()
    print(criterions)
    for criterion in criterions.values():
        criterion.to(device)

    # Metrics
    utils.metrics.myLPIPS.to(device)

    # Optimizer
    parameters = model.parameters()
    optimizer = get_optim(
        parameters=parameters,
        optimizer_name=opt.optimizer,
        lr=opt.lr,
        schedule=opt.lr_schedule,
        weight_decay=opt.weight_decay,
        num_epochs=opt.epoch,
        num_iters_train=num_iters_train,
        cyclic_stepsize=opt.cyclic_step,
        onecycle_pct_start=opt.onecycle_pct_start,
        multistep_milestones=opt.multistep_milestones,
        adam_beta=opt.adam_beta,
    )
    optimizers = {'optimizer': optimizer}
    
    # Load
    resume_ckpt = None
    bare_model = model.module if world_size > 1 else model
    if opt.auto_resume:
        assert opt.resume_ckpt is None
        load_path = os.path.join(
            manager.get_checkpoint_dir(), f'{opt.load_run_ckpt_name}.pt')
        if os.path.exists(load_path):
            opt.resume_ckpt = 'ckpt-last.pt'
    if opt.resume_ckpt:
        load_path = os.path.join(manager.get_checkpoint_dir(), opt.resume_ckpt)
        logger.info(f'==> Resume from checkpoint {load_path}')
        resume_ckpt = torch.load(load_path, map_location='cpu')
    elif opt.load_ckpt or opt.load_run_name:
        if opt.load_ckpt:
            load_path = opt.load_ckpt
        else:
            ckpt_dir = manager.get_checkpoint_dir(
                opt.load_run_name, opt.load_run_number)
            load_path = os.path.join(ckpt_dir, 'ckpt-last.pt')
        logger.info(f'==> Loading model from {load_path}')
        checkpoint = torch.load(load_path, map_location='cpu')
        bare_model.load_state_dict(checkpoint['model'])
    else:
        logger.info(f'==> Will train from scratch')

    # Trainer
    loop_configs = [
        LoopConfig('train', dataset, trainloader,
                   training=True, n_iterations=num_iters_train,
                   n_computation_steps=opt.accum_steps),
        # LoopConfig('val', dataset, valloader,
        #            training=False, n_iterations=num_iters_val,
        #            for_best_meter=True),
        # LoopConfig('test-trainset', dataset, raw_trainloader,
        #            training=False, n_iterations=num_iters_trainset_test,
        #            run_every_n_epochs=opt.trainset_test_period,
        #            run_at_checkpoint=False),
        LoopConfig('test-testset', dataset, testloader,
                   training=False, n_iterations=num_iters_test,
                   run_every_n_epochs=opt.test_period,
                   for_best_meter=True, for_ckpt_meter=True)
    ]
    trainer = StandardTrainer(
        manager=manager,
        models=models,
        criterions=criterions,
        n_epochs=opt.epoch,
        loop_configs=loop_configs,
        optimizers=optimizers,
        log_period=opt.log_period,
        ckpt_period=opt.ckpt_period,
        device=device,
        resume_ckpt=resume_ckpt,
        aligned=opt.aligned,
        alignment_net=opt.alignment_net if opt.aligned else None
    )

    trainer.train()


if __name__ == "__main__":
    # Set the environment variables if not launched by torchrun
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = os.environ['RANK']
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'LOCAL_WORLD_SIZE' not in os.environ:
        os.environ['LOCAL_WORLD_SIZE'] = os.environ['WORLD_SIZE']
    main()
