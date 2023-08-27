import math
import numbers
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from einops import rearrange
from models.common import default_conv, Upsampler, ResBlock, TSAFusion


class PointMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.fc(x)

class SplitPointMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        hidden_dim = int(dim//2 * mlp_ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(dim//2, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim//2, 1, 1, 0),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.fc(x1)
        x = torch.cat([x1, x2], dim=1)
        return rearrange(x, 'b (g d) h w -> b (d g) h w', g=8)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight
        # return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# Shuffle Mixing layer
class SMLayer(nn.Module):
    def __init__(self, dim, kernel_size, mlp_ratio=2):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.spatial = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)

        self.mlp1 = SplitPointMlp(dim, mlp_ratio)
        self.mlp2 = SplitPointMlp(dim, mlp_ratio)

    def forward(self, x):
        x = self.mlp1(self.norm1(x)) + x
        x = self.spatial(x)
        x = self.mlp2(self.norm2(x)) + x
        return x


# Feature Mixing Block
class FMBlock(nn.Module):
    def __init__(self, dim, kernel_size, mlp_ratio=2):
        super().__init__()
        self.net = nn.Sequential(
            SMLayer(dim, kernel_size, mlp_ratio),
            SMLayer(dim, kernel_size, mlp_ratio),
        )
        # self.conv = nn.Sequential(
        #     nn.Conv2d(dim, dim + 16, 3, 1, 1),
        #     nn.GELU(),
        #     nn.Conv2d(dim + 16, dim, 1, 1, 0)
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim * mlp_ratio, dim, 1, 1, 0)
        )

    def forward(self, x):
        x = self.net(x) + x
        x = self.conv(x) + x
        return x


class ShuffleMixer(nn.Module):
    """
    Args:
        n_feats (int): Number of channels. Default: 64 (32 for the tiny model).
        kerenl_size (int): kernel size of Depthwise convolution. Default:7 (3 for the tiny model).
        n_blocks (int): Number of feature mixing blocks. Default: 5.
        mlp_ratio (int): The expanding factor of point-wise MLP. Default: 2.
        upscaling_factor: The upscaling factor. [2, 3, 4]
    """
    def __init__(self, dim=64, burst_size=14, in_channel=3, out_channel=3, scale=4, kernel_size=7, n_blocks=8, mlp_ratio=2):
        super().__init__()
        self.dim = dim
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale = scale

        conv = default_conv
        self.head = conv(in_channel, dim, 3)
        self.body = nn.Sequential(*[ResBlock(conv, dim, 3) for _ in range(2)])
        self.fusion = TSAFusion(num_feat=dim, num_frame=burst_size)
        

        self.blocks = nn.Sequential(
            *[FMBlock(dim, kernel_size, mlp_ratio) for _ in range(n_blocks)]
        )

        self.upsample = Upsampler(default_conv, scale, dim)

        self.conv_last = nn.Conv2d(dim, self.out_channel,
                                   kernel_size=3, stride=1, padding=1)

    def forward(self, x, x_base=None):
        b, t, c, h, w = x.size()
        assert t == 14, 'Frame should be 14!'
        assert c == self.in_channel, f'In channels should be {self.in_channel}!'

        if x_base is None:
            x_base = x[:, 0, :, :, :].contiguous()
            x_base_scale = self.scale
        else:
            x_base_scale = self.scale // 2

        x_feat_head = self.head(x.view(-1, c, h, w))  # [b*t, dim, h, w]
        x_feat_body = self.body(x_feat_head)  # [b*t, dim, h, w]
        feat = x_feat_body.view(b, t, -1, h, w)   # [b, t, dim, h, w]
        fusion_feat = self.fusion(feat)   # fusion feat [b, dim, h, w]

        y = self.blocks(fusion_feat)

        y = self.upsample(y)
        out = self.conv_last(y)

        base = F.interpolate(x_base, scale_factor=x_base_scale,
                             mode='bilinear', align_corners=False)
        out = base + out

        return out

if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # 720p [1280 * 720]
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    # x = torch.randn(1, 3, 320, 180)
    x = torch.randn(1, 14, 3, 160, 160)

    # model = ShuffleMixer(dim=64, kernel_size=21, n_blocks=5, mlp_ratio=2)
    model = ShuffleMixer(dim=64, kernel_size=7, n_blocks=8, mlp_ratio=2)
    print(model)
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(f'output: {output.shape}')
