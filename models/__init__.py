from .unet import UNet
from .uformer import Uformer, LeFF


__all__ = ['get_model']


def get_model(arch, dim, burst_size, in_channel, scale):
    if arch.startswith('unet'):
        model = UNet(dim=dim, burst_size=burst_size,
                     in_channel=in_channel, scale=scale, conv_block=arch[5:])
    elif arch.startswith('uformer'):
        mlp_arch = arch[8:]
        if mlp_arch.startswith('leff'):
            model = Uformer(mlps=LeFF, in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
        else:
            model = Uformer(in_chans=in_channel, scale=scale, embed_dim=dim, burst_size=burst_size)
    else:
        raise NotImplementedError(f"Unknown arch: {arch}")
    return model
