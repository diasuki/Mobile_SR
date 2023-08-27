from .unet import UNet, UNet_student
from functools import partial

__all__ = ['get_model']


def get_model(arch, dim, burst_size, in_channel, scale):
    if arch.startswith('unet'):
        model = UNet(dim=dim, burst_size=burst_size,
                     in_channel=in_channel, scale=scale, conv_block=arch[5:])
    elif arch.startswith('swinir'):
        from .swinir import SwinIR
        model = SwinIR(embed_dim=dim, burst_size=burst_size, in_chans=in_channel, scale=scale)
    elif arch.startswith('shufflemixer'):
        from .shufflemixer_arch import ShuffleMixer
        model = ShuffleMixer(dim=dim, burst_size=burst_size, in_channel=in_channel, scale=scale)
    elif arch.startswith('safmn'):
        if arch == 'safmn1':
            from .safmn_arch import SAFMN1
            model = SAFMN1(dim=dim, burst_size=burst_size, in_channel=in_channel, scale=scale)
        elif arch == 'safmn':
            from .safmn_arch import SAFMN
            model = SAFMN(dim=dim, burst_size=burst_size, in_channel=in_channel, scale=scale)
        else:
            raise NotImplementedError(f"Unknown arch: {arch}")
    elif arch.startswith('uformer2_tiny'):
        from .uformer2_tiny import Uformer, LeFF, Anti_FFN
        from .uformer2_tiny import SepConv
        kernel_size = arch.rsplit('_')[-1]
        kernel_size = int(kernel_size) if kernel_size.isnumeric() else 3
        mlp_arch = arch[14:]
        if mlp_arch.startswith('leff'):
            model = Uformer(token_mixers=partial(SepConv, kernel_size=kernel_size, padding=kernel_size//2), mlps=LeFF, in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
        elif mlp_arch.startswith('anti'):
            model = Uformer(token_mixers=partial(SepConv, kernel_size=kernel_size, padding=kernel_size//2), mlps=Anti_FFN, in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
        else:
            model = Uformer(token_mixers=partial(SepConv, kernel_size=kernel_size, padding=kernel_size//2), in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
    elif arch.startswith('uformer1_tiny'):
        from .uformer1_tiny import Uformer, LeFF, Anti_FFN
        from .uformer1_tiny import SepConv
        kernel_size = arch.rsplit('_')[-1]
        kernel_size = int(kernel_size) if kernel_size.isnumeric() else 3
        mlp_arch = arch[14:]
        if mlp_arch.startswith('leff'):
            model = Uformer(token_mixers=partial(SepConv, kernel_size=kernel_size, padding=kernel_size//2), mlps=LeFF, in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
        elif mlp_arch.startswith('anti'):
            model = Uformer(token_mixers=partial(SepConv, kernel_size=kernel_size, padding=kernel_size//2), mlps=Anti_FFN, in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
        else:
            model = Uformer(token_mixers=partial(SepConv, kernel_size=kernel_size, padding=kernel_size//2), in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
    elif arch.startswith('uformer_tiny'):
        from .uformer_tiny import Uformer, LeFF, Anti_FFN
        from .uformer_tiny import SepConv
        kernel_size = arch.rsplit('_')[-1]
        kernel_size = int(kernel_size) if kernel_size.isnumeric() else 3
        mlp_arch = arch[13:]
        if mlp_arch.startswith('leff'):
            model = Uformer(token_mixers=partial(SepConv, kernel_size=kernel_size, padding=kernel_size//2), mlps=LeFF, in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
        elif mlp_arch.startswith('anti'):
            model = Uformer(token_mixers=partial(SepConv, kernel_size=kernel_size, padding=kernel_size//2), mlps=Anti_FFN, in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
        else:
            model = Uformer(token_mixers=partial(SepConv, kernel_size=kernel_size, padding=kernel_size//2), in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
    elif arch.startswith('uformer'):
        from .uformer import Uformer, LeFF
        mlp_arch = arch[8:]
        if mlp_arch.startswith('leff'):
            model = Uformer(mlps=LeFF, in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
        elif mlp_arch.startswith('anti'):
            model = Uformer(mlps=Anti_FFN, in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
        else:
            model = Uformer(in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
    else:
        raise NotImplementedError(f"Unknown arch: {arch}")
    return model


def get_student_model(arch, dim, burst_size, in_channel, scale):
    if arch.startswith('unet'):
        model = UNet_student(dim=dim, burst_size=burst_size,
                     in_channel=in_channel, scale=scale, conv_block=arch[5:])
    else:
        raise NotImplementedError(f"Unknown arch: {arch}")
    return model


def get_teacher_model(mask_ratio=0.):
    from .basemodel import Uformer
    model = Uformer(mask_ratio=mask_ratio)
    return model


def get_teacher_model_distill(mask_ratio=0.):
    from .basemodel import Uformer_teacher
    model = Uformer_teacher(mask_ratio=mask_ratio)
    return model