import torch
import torch.nn as nn
import torch.nn.functional as F
import contextual_loss as cl


class L1_with_CoBi(nn.Module):
    def __init__(self, boundary_ignore=None, lamda=1.0):
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.lamda = lamda
        self.cobi_loss = cl.ContextualBilateralLoss(use_vgg=True, vgg_layer='relu5_4').cuda()

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        pred_m = pred
        gt_m = gt

        if valid is None:
            mse = F.l1_loss(pred_m, gt_m) + self.lamda * self.cobi_loss(pred_m, gt_m)
        else:
            assert False, "L1_with_CoBi without valid input!"
            mse = F.l1_loss(pred_m, gt_m, reduction='none')

            eps = 1e-12
            elem_ratio = mse.numel() / valid.numel()
            mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse
