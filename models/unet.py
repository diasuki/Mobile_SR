import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import default_conv, Upsampler, ResBlock, TSAFusion

import math

# class Upsample(nn.Module):

#     def __init__(self, channel):
#         super().__init__()
#         self.conv = nn.Conv2d(channel, channel * 4, 3, padding=1, bias=True)
#         self.pixel_shuffle = nn.PixelShuffle(2)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.pixel_shuffle(x)
#         return x


class ResBlock2(nn.Module):

    def __init__(self, in_channel, out_channel, strides=1, first=False):
        super().__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel,
                                kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

'''
something wrong with [inplace=True]
'''
class PreActResBlock4(nn.Module):

    def __init__(self, in_channel, out_channel, strides=1, first=False):
        super().__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block1 = nn.Sequential(
            nn.LeakyReLU(inplace=True) if not first else nn.Identity(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
        )
        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel,
                                      kernel_size=1, stride=strides, padding=0)
        else:
            self.shortcut = nn.Identity()
        self.block2 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
        )

    def forward(self, x):
        out1 = self.block1(x)
        res1 = self.shortcut(x)
        x = out1 + res1
        out2 = self.block2(x)
        res2 = x
        return out2 + res2


'''
start with [inplace=False] for some blocks
'''
class PreActResBlock4_new(nn.Module):

    def __init__(self, in_channel, out_channel, strides=1, first=False):
        super().__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block1 = nn.Sequential(
            nn.LeakyReLU(inplace=False) if not first else nn.Identity(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
        )
        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel,
                                      kernel_size=1, stride=strides, padding=0)
        else:
            self.shortcut = nn.Identity()
        self.block2 = nn.Sequential(
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
        )

    def forward(self, x):
        out1 = self.block1(x)
        res1 = self.shortcut(x)
        x = out1 + res1
        out2 = self.block2(x)
        res2 = x
        return out2 + res2


class PreActBNResBlock4(nn.Module):

    def __init__(self, in_channel, out_channel, strides=1, first=False):
        super().__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_channel) if not first else nn.Identity(),
            nn.LeakyReLU(inplace=True) if not first else nn.Identity(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
        )
        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel,
                                      kernel_size=1, stride=strides, padding=0)
        else:
            self.shortcut = nn.Identity()
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
        )

    def forward(self, x):
        out1 = self.block1(x)
        res1 = self.shortcut(x)
        x = out1 + res1
        out2 = self.block2(x)
        res2 = x
        return out2 + res2


class UNet(nn.Module):

    def __init__(self, dim=64, burst_size=14, in_channel=3, out_channel=3, scale=4, conv_block='res2'):
        super(UNet, self).__init__()

        self.dim = dim
        self.in_channel = in_channel
        # self.out_channel = in_channel if out_channel is None else out_channel
        self.out_channel = out_channel
        self.scale = scale

        conv = default_conv
        if conv_block == 'pares4':
            ConvBlock = PreActResBlock4
        elif conv_block == 'pares4_new':
            ConvBlock = PreActResBlock4_new
        elif conv_block == 'pares4_bn':
            ConvBlock = PreActBNResBlock4
        elif conv_block == 'res2':
            ConvBlock = ResBlock2
        else:
            raise NotImplementedError()

        self.head = conv(in_channel, dim, 3)
        self.body = nn.Sequential(*[ResBlock(conv, dim, 3) for _ in range(2)])
        # self.body = nn.Identity()
        self.fusion = TSAFusion(num_feat=dim, num_frame=burst_size)

        self.ConvBlock1 = ConvBlock(dim, dim, strides=1, first=True)
        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = ConvBlock(dim, dim, strides=1)
        self.pool2 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = ConvBlock(dim, dim, strides=1)
        self.pool3 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = ConvBlock(dim, dim, strides=1)
        self.pool4 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = ConvBlock(dim, dim, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.ConvBlock6 = ConvBlock(dim*2, dim, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.ConvBlock7 = ConvBlock(dim*2, dim, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.ConvBlock8 = ConvBlock(dim*2, dim, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.ConvBlock9 = ConvBlock(dim*2, dim, strides=1)

        # self.ConvBlock10 = ConvBlock(dim, dim, strides=1)
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

        conv1 = self.ConvBlock1(fusion_feat)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        up10 = self.upsample(conv9)
        out = self.conv_last(up10)

        base = F.interpolate(x_base, scale_factor=x_base_scale,
                             mode='bilinear', align_corners=False)
        out = base + out

        return out

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        else:
            raise NotImplementedError(f"Unknown module: {m}")

    return nn.Sequential(*C)

class UNet_student(nn.Module):

    def __init__(self, dim=64, burst_size=14, in_channel=3, out_channel=3, scale=4, conv_block='res2'):
        super(UNet_student, self).__init__()

        t_channels = [dim * 16, dim * 8, dim * 4, dim * 2]
        s_channels = [dim] * 4
        self.s_channels = s_channels
        self.connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        self.dim = dim
        self.in_channel = in_channel
        # self.out_channel = in_channel if out_channel is None else out_channel
        self.out_channel = out_channel
        self.scale = scale

        conv = default_conv
        if conv_block == 'pares4':
            ConvBlock = PreActResBlock4
        elif conv_block == 'pares4_new':
            ConvBlock = PreActResBlock4_new
        elif conv_block == 'pares4_bn':
            ConvBlock = PreActBNResBlock4
        elif conv_block == 'res2':
            ConvBlock = ResBlock2
        else:
            raise NotImplementedError()

        self.head = conv(in_channel, dim, 3)
        self.body = nn.Sequential(*[ResBlock(conv, dim, 3) for _ in range(2)])
        # self.body = nn.Identity()
        self.fusion = TSAFusion(num_feat=dim, num_frame=burst_size)

        self.ConvBlock1 = ConvBlock(dim, dim, strides=1, first=True)
        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = ConvBlock(dim, dim, strides=1)
        self.pool2 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = ConvBlock(dim, dim, strides=1)
        self.pool3 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = ConvBlock(dim, dim, strides=1)
        self.pool4 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = ConvBlock(dim, dim, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.ConvBlock6 = ConvBlock(dim*2, dim, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.ConvBlock7 = ConvBlock(dim*2, dim, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.ConvBlock8 = ConvBlock(dim*2, dim, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.ConvBlock9 = ConvBlock(dim*2, dim, strides=1)

        # self.ConvBlock10 = ConvBlock(dim, dim, strides=1)
        self.upsample = Upsampler(default_conv, scale, dim)

        self.conv_last = nn.Conv2d(dim, self.out_channel,
                                   kernel_size=3, stride=1, padding=1)

    def forward(self, x, x_base=None):
        feature_maps = []
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

        conv1 = self.ConvBlock1(fusion_feat)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)
        feature_maps.append(conv6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)
        feature_maps.append(conv7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)
        feature_maps.append(conv8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)
        feature_maps.append(conv9)

        up10 = self.upsample(conv9)
        out = self.conv_last(up10)

        base = F.interpolate(x_base, scale_factor=x_base_scale,
                             mode='bilinear', align_corners=False)
        out = base + out

        # expand channels of student feature
        for i in range(len(self.s_channels)):
            feature_maps[i] = self.connectors[i](feature_maps[i])

        return feature_maps, out