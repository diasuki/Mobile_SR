import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from models.common import default_conv, Upsampler, ResBlock, TSAFusion


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# SE
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


# Channel MLP: Conv1*1 -> Conv1*1
class ChannelMLP(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mlp(x)


# MBConv: Conv1*1 -> DW Conv3*3 -> [SE] -> Conv1*1
class MBConv(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mbconv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mbconv(x)


# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
        
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        
        # Activation
        self.act = nn.GELU() 

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**i, w//2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        # Multiscale Block
        self.safm = SAFM(dim) 
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale) 

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x
        
        
class SAFMN(nn.Module):
    def __init__(self, dim=64, burst_size=14, in_channel=3, out_channel=3, scale=4, n_blocks=8, ffn_scale=2.0):
        super().__init__()
        self.dim = dim
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale = scale

        conv = default_conv
        self.head = conv(in_channel, dim, 3)
        self.body = nn.Sequential(*[ResBlock(conv, dim, 3) for _ in range(2)])
        self.fusion = TSAFusion(num_feat=dim, num_frame=burst_size)

        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

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

        y = self.feats(fusion_feat) + fusion_feat

        y = self.upsample(y)
        out = self.conv_last(y)

        base = F.interpolate(x_base, scale_factor=x_base_scale,
                             mode='bilinear', align_corners=False)

        out = base + out

        return out


class SAFMN1(nn.Module):
    def __init__(self, dim=64, burst_size=14, in_channel=3, out_channel=3, scale=4, n_blocks=8, ffn_scale=2.0):
        super().__init__()
        self.dim = dim
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale = scale

        conv = default_conv
        self.head = conv(in_channel, dim, 3)
        self.body = nn.Sequential(*[ResBlock(conv, dim, 3) for _ in range(2)])
        self.fusion = TSAFusion(num_feat=dim, num_frame=burst_size)

        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

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

        y = self.feats(fusion_feat) + fusion_feat

        y = self.upsample(y)
        out = self.conv_last(y)

        # base = F.interpolate(x_base, scale_factor=x_base_scale,
        #                      mode='bilinear', align_corners=False)

        # out = base + out

        return out


if __name__== '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 14, 3, 160, 160)
    # x = torch.randn(1, 3, 256, 256)

    model = SAFMN(dim=64, burst_size=14, in_channel=3, out_channel=3, scale=4, n_blocks=8, ffn_scale=2.0)
    # model = SAFMN(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)
