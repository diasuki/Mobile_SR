import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils.experiman import manager
from data import *
from models import get_model
from trainers import StandardTester, LoopConfig
from utils.misc import parse
import utils.metrics
from main_standard import add_parser_argument
import datetime


def add_parser_argument_eval(parser):
    ## ====================== Logging =========================
    parser.add_argument('--save_images', action='store_true')
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
    add_parser_argument_eval(parser)
    opt = parser.parse_args()
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
        world_size=world_size, rank=rank)
    testloader = dataset.get_loader(split='val', **data_kwargs)
    num_iters_test = parse(opt.num_iters_test, len(testloader))

    # Model
    model = get_model(
        arch=opt.arch, dim=opt.dim, burst_size=opt.burst_size,
        in_channel=opt.in_channel, scale=opt.scale,
    ).to(device)
    model = torch.compile(model)
    # model = torch.compile(model, mode='reduce-overhead')
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    models = {'model': model}

    # Aligned
    if opt.aligned:
        from pwcnet.pwcnet import PWCNet
        alignment_net = PWCNet(load_pretrained=True,
                           weights_path='./pwcnet/pwcnet-network-default.pth')
        alignment_net = alignment_net.to('cuda')
        for param in alignment_net.parameters():
            param.requires_grad = False
        opt.alignment_net = alignment_net

    # Metrics
    utils.metrics.myLPIPS.to(device)
    
    # Load
    bare_model = model.module if world_size > 1 else model
    if opt.load_ckpt:
        load_path = opt.load_ckpt
    else:
        ckpt_dir = manager.get_checkpoint_dir(
            opt.load_run_name, opt.load_run_number)
        load_path = os.path.join(ckpt_dir, f'{opt.load_run_ckpt_name}.pt')
    logger.info(f'==> Loading model from {load_path}')
    checkpoint = torch.load(load_path, map_location='cpu')
    bare_model.load_state_dict(checkpoint['model'])

    # Trainer
    loop_configs = [
        LoopConfig('test-testset', dataset, testloader,
                   training=False, n_iterations=num_iters_test)
    ]
    trainer = StandardTester(
        manager=manager,
        models=models,
        criterions={},
        n_epochs=1,
        loop_configs=loop_configs,
        optimizers={},
        log_period=opt.log_period,
        ckpt_period=opt.ckpt_period,
        device=device,
        aligned=opt.aligned,
        alignment_net=opt.alignment_net if opt.aligned else None
    )

    trainer.test()


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
