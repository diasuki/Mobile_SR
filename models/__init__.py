from .unet import UNet


__all__ = ['get_model']


def get_model(arch, dim, burst_size, in_channel, scale):
    if arch.startswith('unet'):
        model = UNet(dim=dim, burst_size=burst_size,
                     in_channel=in_channel, scale=scale, conv_block=arch[5:])
    else:
        raise NotImplementedError(f"Unknown arch: {arch}")
    return model
