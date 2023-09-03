import math
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=0):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)

        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


class Downsample_flatten(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample_flatten, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        # import pdb;pdb.set_trace()
        out = self.conv(x).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H/2*W/2*self.in_channel*self.out_channel*4*4
        print("Downsample:{%.2f}"%(flops/1e9))
        return flops

class Upsample_flatten(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample_flatten, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.deconv(x).contiguous() # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops


############ New Fusion Block ###############
class NewFusion(nn.Module):
    def __init__(self, num_feat=64, num_frame=14, center_frame_idx=0):
        super(NewFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention
        self.downsample1 = Downsample_flatten(num_feat, num_feat*2)
        self.downsample2 = Downsample_flatten(num_feat*2, num_feat*4)

        self.upsample1 = Upsample_flatten(num_feat*4, num_feat*2)
        self.upsample2 = Upsample_flatten(num_feat*4, num_feat)

        n_resblocks = 2
        conv = default_conv
        m_res_block1 = [
            ResBlock(
                conv, num_feat, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_res_block2 = [
            ResBlock(
                conv, num_feat*2, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_res_block3 = [
            ResBlock(
                conv, num_feat*4, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_res_block4 = [
            ResBlock(
                conv, num_feat*4, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_res_block5 = [
            ResBlock(
                conv, num_feat*2, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_fusion_tail = [conv(num_feat*2, num_feat, kernel_size=3)]

        self.res_block1 = nn.Sequential(*m_res_block1)
        self.res_block2 = nn.Sequential(*m_res_block2)
        self.res_block3 = nn.Sequential(*m_res_block3)
        self.res_block4 = nn.Sequential(*m_res_block4)
        self.res_block5 = nn.Sequential(*m_res_block5)
        self.fusion_tail = nn.Sequential(*m_fusion_tail)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, aligned_feat):
        b, t, c, h, w = aligned_feat.size()

        # attention map, highlight similarities while keep similarities
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # [b,t,c,h,w]

        corr_diff = []
        corr_l = []
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1).unsqueeze(1)  # [b,1,h,w]
            corr_l.append(corr)
            if i == 0:
                corr_diff.append(corr)  # [b,1,h,w]
            else:
                corr_difference = torch.abs(corr_l[i] - corr_l[i-1])
                corr_diff.append(corr_difference)
                
                
        corr_l_cat = torch.cat(corr_l, dim=1)
        corr_prob = torch.sigmoid(torch.cat(corr_diff, dim=1))  # [b,t,h,w]
                
        corr_l_cat = torch.cat(corr_l, dim=1)
        corr_prob = torch.sigmoid(torch.cat(corr_diff, dim=1))  # [b,t,h,w]

        
#         transform_tensor = transforms.ToTensor()
#         transform_pil = transforms.ToPILImage()

#         for b_i in range(b):
#             for t_i in range(1, t):
#                 corr_prob_i = corr_prob[b_i, t_i, :, :]
#                 corr_prob_img = transform_tensor(transform_pil(corr_prob_i).convert('L')).squeeze(0)
#                 draw(corr_prob_img, burst_name[b_i] + '_{}_diff.png'.format(t_i))

#         for b_i in range(b):
#             for t_i in range(1, t):
#                 corr_l_cat_i = corr_l_cat[b_i, t_i, :, :]
#                 corr_l_cat_img = transform_tensor(transform_pil(corr_l_cat_i).convert('L')).squeeze(0)
#                 draw(corr_l_cat_img, burst_name[b_i] + '_{}_affinity.png'.format(t_i))

        
# # #         corr_prob = torch.cat(corr_diff, dim=1)
        
# # #         for corr_index in range(t):
# # #             print("corr_prob[{}].max: {}, min: {}".format(corr_index, corr_prob[:, corr_index, :, :].max(), corr_prob[:, corr_index, :, :].min()))
        
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)  # [b,t,c,h,w]
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # [b,t*c,h,w]

        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        feat = self.lrelu(self.feat_fusion(aligned_feat))  # [b,c,h,w]

        # Hourglass for spatial attention
        feat_res1 = self.res_block1(feat)
        down_feat1 = self.downsample1(feat_res1)
        feat_res2 = self.res_block2(down_feat1)
        down_feat2 = self.downsample2(feat_res2)

        feat3 = self.res_block3(down_feat2)

        up_feat3 = self.upsample1(feat3)
        concat_2_1 = torch.cat([up_feat3, feat_res2], 1)
        feat_res4 = self.res_block4(concat_2_1)
        up_feat4 = self.upsample2(feat_res4)
        concat_1_0 = torch.cat([up_feat4, feat_res1], 1)
        feat_res5 = self.res_block5(concat_1_0)

        feat_out = self.fusion_tail(feat_res5) + feat

        return feat_out