"""
## Uformer: A General U-Shaped Transformer for Image Restoration
## Zhendong Wang, Xiaodong Cun, Jianmin Bao, Jianzhuang Liu
## https://arxiv.org/abs/2106.03106
"""
import cv2
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
from models.common import default_conv, ResBlock, Upsampler, TSAFusion
from functools import partial


# class FastLeFF(nn.Module):
    
#     def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
#         super().__init__()

#         from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

#         self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
#                                 act_layer())
#         self.dwconv = nn.Sequential(DepthwiseConv2d(hidden_dim, hidden_dim, kernel_size=3,stride=1,padding=1),
#                         act_layer())
#         self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
#         self.dim = dim
#         self.hidden_dim = hidden_dim

#     def forward(self, x):
#         # bs x hw x c
#         bs, hw, c = x.size()
#         hh = int(math.sqrt(hw))

#         x = self.linear1(x)

#         # spatial restore
#         x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
#         # bs,hidden_dim,32x32

#         x = self.dwconv(x)

#         # flaten
#         x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

#         x = self.linear2(x)

#         return x

#     def flops(self, H, W):
#         flops = 0
#         # fc1
#         flops += H*W*self.dim*self.hidden_dim 
#         # dwconv
#         flops += H*W*self.hidden_dim*3*3
#         # fc2
#         flops += H*W*self.hidden_dim*self.dim
#         print("LeFF:{%.2f}"%(flops/1e9))
#         return flops


class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size =k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self): 
        flops = 0
        flops += self.channel*self.channel*self.k_size
        
        return flops


class LeFF(nn.Module):
    def __init__(self, dim, mlp_ratio=4, act_layer=nn.GELU, drop = 0., use_eca=False):
        super().__init__()
        hidden_dim = int(mlp_ratio * dim)
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.dim*self.hidden_dim 
        # dwconv
        flops += H*W*self.hidden_dim*3*3
        # fc2
        flops += H*W*self.hidden_dim*self.dim
        print("LeFF:{%.2f}"%(flops/1e9))
        # eca 
        if hasattr(self.eca, 'flops'): 
            flops += self.eca.flops()
        return flops


#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H/2*W/2*self.in_channel*self.out_channel*4*4
        print("Downsample:{%.2f}"%(flops/1e9))
        return flops

# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops

# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        print("Input_proj:{%.2f}"%(flops/1e9))
        return flops

# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        print("Output_proj:{%.2f}"%(flops/1e9))
        return flops


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale
        

class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RandomMixing(nn.Module):
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1), 
            requires_grad=False)
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x


class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default. 
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance. 
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True, 
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster, 
    because it directly utilizes otpimized F.layer_norm
    """
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity, 
        # bias=False, kernel_size=7, padding=3,
        bias=True, kernel_size=3, padding=1,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.view(B, H, W, C)

        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)

        x = x.view(B, H * W, C)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfiled for [B, H, W, C] input
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
        norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        
    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


class Uformer(nn.Module):
    def __init__(self, in_chans=3, scale=4, burst_size=14,
                 embed_dim=64, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 token_mixers=SepConv, mlps=Mlp,
                 norm_layers=partial(LayerNormWithoutBias, eps=1e-6), # partial(LayerNormGeneral, eps=1e-6, bias=False),
                 drop_path_rate=0.,
                 layer_scale_init_values=None,
                #  res_scale_init_values=[None, None, 1.0, 1.0, 1.0, 1.0, 1.0, None, None],
                 res_scale_init_values=None,
                 drop_rate=0.,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        
        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage
        
        self.scale = scale
        self.embed_dim = embed_dim
        self.token_mixers = token_mixers
        self.mlps = mlps
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.num_frames = burst_size

        # Input/Output
        self.input_proj = InputProj(in_channel=embed_dim, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=embed_dim, kernel_size=3, stride=1)

        conv = default_conv
        n_resblocks = 2

        m_head = [conv(in_chans, embed_dim, kernel_size=3)]

        m_body = [
            ResBlock(
                conv, embed_dim, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        self.fusion = TSAFusion(num_feat=embed_dim, num_frame=self.num_frames)

        m_tail = [
            Upsampler(conv, scale, embed_dim, act=False),
            conv(embed_dim, in_chans, kernel_size=3)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        # Encoder
        self.encoderlayer_0 = nn.Sequential(
                *[MetaFormerBlock(dim=embed_dim,
                token_mixer=token_mixers[0],
                mlp=mlps[0],
                norm_layer=norm_layers[0],
                drop_path=dp_rates[0 + j],
                layer_scale_init_value=layer_scale_init_values[0],
                res_scale_init_value=res_scale_init_values[0],
                ) for j in range(depths[0])]
            )
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = nn.Sequential(
                *[MetaFormerBlock(dim=embed_dim*2,
                token_mixer=token_mixers[1],
                mlp=mlps[1],
                norm_layer=norm_layers[1],
                drop_path=dp_rates[1 + j],
                layer_scale_init_value=layer_scale_init_values[1],
                res_scale_init_value=res_scale_init_values[1],
                ) for j in range(depths[1])]
            )
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)
        self.encoderlayer_2 = nn.Sequential(
                *[MetaFormerBlock(dim=embed_dim*4,
                token_mixer=token_mixers[2],
                mlp=mlps[2],
                norm_layer=norm_layers[2],
                drop_path=dp_rates[2 + j],
                layer_scale_init_value=layer_scale_init_values[2],
                res_scale_init_value=res_scale_init_values[2],
                ) for j in range(depths[2])]
            )

        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)
        self.encoderlayer_3 = nn.Sequential(
                *[MetaFormerBlock(dim=embed_dim*8,
                token_mixer=token_mixers[3],
                mlp=mlps[3],
                norm_layer=norm_layers[3],
                drop_path=dp_rates[3 + j],
                layer_scale_init_value=layer_scale_init_values[3],
                res_scale_init_value=res_scale_init_values[3],
                ) for j in range(depths[3])]
            )
        self.dowsample_3 = dowsample(embed_dim*8, embed_dim*16)

        # Bottleneck
        self.conv = nn.Sequential(
                *[MetaFormerBlock(dim=embed_dim*16,
                token_mixer=token_mixers[4],
                mlp=mlps[4],
                norm_layer=norm_layers[4],
                drop_path=dp_rates[4 + j],
                layer_scale_init_value=layer_scale_init_values[4],
                res_scale_init_value=res_scale_init_values[4],
                ) for j in range(depths[4])]
            )

        # Decoder
        self.upsample_0 = upsample(embed_dim*16, embed_dim*8)
        self.decoderlayer_0 = nn.Sequential(
                *[MetaFormerBlock(dim=embed_dim*16,
                token_mixer=token_mixers[5],
                mlp=mlps[5],
                norm_layer=norm_layers[5],
                drop_path=dp_rates[5 + j],
                layer_scale_init_value=layer_scale_init_values[5],
                res_scale_init_value=res_scale_init_values[5],
                ) for j in range(depths[5])]
            )
        self.upsample_1 = upsample(embed_dim*16, embed_dim*4)
        self.decoderlayer_1 = nn.Sequential(
                *[MetaFormerBlock(dim=embed_dim*8,
                token_mixer=token_mixers[6],
                mlp=mlps[6],
                norm_layer=norm_layers[6],
                drop_path=dp_rates[6 + j],
                layer_scale_init_value=layer_scale_init_values[6],
                res_scale_init_value=res_scale_init_values[6],
                ) for j in range(depths[6])]
            )
        self.upsample_2 = upsample(embed_dim*8, embed_dim*2)
        self.decoderlayer_2 = nn.Sequential(
                *[MetaFormerBlock(dim=embed_dim*4,
                token_mixer=token_mixers[7],
                mlp=mlps[7],
                norm_layer=norm_layers[7],
                drop_path=dp_rates[7 + j],
                layer_scale_init_value=layer_scale_init_values[7],
                res_scale_init_value=res_scale_init_values[7],
                ) for j in range(depths[7])]
            )
        self.upsample_3 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_3 = nn.Sequential(
                *[MetaFormerBlock(dim=embed_dim*2,
                token_mixer=token_mixers[8],
                mlp=mlps[8],
                norm_layer=norm_layers[8],
                drop_path=dp_rates[8 + j],
                layer_scale_init_value=layer_scale_init_values[8],
                res_scale_init_value=res_scale_init_values[8],
                ) for j in range(depths[8])]
            )

        self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_mixers={self.token_mixers}, mlps={self.mlps}"

    def forward(self, x, x_base=None):
        # Input Multi-Frame Conv
        b, t, c, h, w = x.size()
        assert t == 14, 'Frame should be 14!'
        assert c == 3, 'In channels should be 3!'

        if x_base is None:
            x_base = x[:, 0, :, :, :].contiguous()
            x_base_scale = self.scale
        else:
            x_base_scale = self.scale // 2

        x_feat_head = self.head(x.view(-1, c, h, w))  # [b*t, embed_dim, h, w]
        x_feat_body = self.body(x_feat_head)  # [b*t, embed_dim, h, w]

        feat = x_feat_body.view(b, t, -1, h, w)   # [b, t, embed_dim, h, w]
        fusion_feat = self.fusion(feat)   # fusion feat [b, embed_dim, h, w]

        assert fusion_feat.dim() == 4, 'Fusion Feat should be [B,C,H,W]!'

        # Input Projection
        y = self.input_proj(fusion_feat)   # B, H*W, C
        y = self.pos_drop(y)

        #Encoder
        conv0 = self.encoderlayer_0(y)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0)
        pool1 = self.dowsample_1(conv1)

        conv2 = self.encoderlayer_2(pool1)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2)
        pool3 = self.dowsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3)

        #Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0,conv3],-1)
        deconv0 = self.decoderlayer_0(deconv0)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1,conv2],-1)
        deconv1 = self.decoderlayer_1(deconv1)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2,conv1],-1)
        deconv2 = self.decoderlayer_2(deconv2)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3,conv0],-1)
        deconv3 = self.decoderlayer_3(deconv3)

        # Output Projection
        y = self.output_proj(deconv3)

        output = self.tail(y)

        base = F.interpolate(x_base, scale_factor=x_base_scale, mode='bilinear', align_corners=False)

        out = output + base

        return out

    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso,self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops()+self.dowsample_0.flops(self.reso,self.reso)
        flops += self.encoderlayer_1.flops()+self.dowsample_1.flops(self.reso//2,self.reso//2)
        flops += self.encoderlayer_2.flops()+self.dowsample_2.flops(self.reso//2**2,self.reso//2**2)
        flops += self.encoderlayer_3.flops()+self.dowsample_3.flops(self.reso//2**3,self.reso//2**3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso//2**4,self.reso//2**4)+self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso//2**3,self.reso//2**3)+self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso//2**2,self.reso//2**2)+self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso//2,self.reso//2)+self.decoderlayer_3.flops()

        # Output Projection
        flops += self.output_proj.flops(self.reso,self.reso)
        return flops

if __name__ == "__main__":
    # arch = Uformer
    # input_size = 256
    # # arch = Uformer_Cross
    # depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
    # # model_restoration = UNet(dim=32)
    # model_restoration = arch(img_size=input_size, embed_dim=44,depths=depths,
    #              win_size=8, mlp_ratio=4., qkv_bias=True,
    #              token_projection='linear', token_mlp='leff',
    #              downsample=Downsample, upsample=Upsample)
    model = Uformer()
    x = torch.randn(1, 14, 3, 16, 16)
    y = model(x)
    print(y.shape)
    # arch = LeWinformer
    # depth = 20
    # model_restoration = arch(embed_dim=16,depth=depth,
    #              win_size=8, mlp_ratio=4., qkv_bias=True,
    #              token_projection='linear', token_mlp='leff',se_layer=False)
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
    #                                             print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print("number of GFLOPs: %.2f G"%(model_restoration.flops(input_size,input_size) / 1e9))
    # print("number of GFLOPs: %.2f G"%(model_restoration.flops() / 1e9))