import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.warp import warp
from losses.gw_loss import GWLoss
from losses.charbonnier import CharbonnierLoss


class AlignedL1(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

    def forward(self, pred, gt):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped_m = warp(pred, flow)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        pred_warped_m = pred_warped_m.contiguous()
        gt = gt.contiguous()
        # Estimate MSE
        l1 = F.l1_loss(pred_warped_m, gt)

        return l1


class AlignedGWLoss(nn.Module):
    def __init__(self, alignment_net, aligned_gw_loss_weight=3.0, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net
        self.charb_loss = CharbonnierLoss()
        self.gw_loss = GWLoss()
        self.aligned_gw_loss_weight = aligned_gw_loss_weight

    def forward(self, pred, gt):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped_m = warp(pred, flow)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        pred_warped_m = pred_warped_m.contiguous()
        gt = gt.contiguous()
        # Estimate MSE
        # l1 = F.l1_loss(pred_warped_m, gt)
        loss = self.charb_loss(pred_warped_m, gt) + self.aligned_gw_loss_weight * self.gw_loss(pred_warped_m, gt)

        return loss