import torch
import lpips
import torch.nn as nn

class LPIPS_loss(nn.Module):
    def __init__(self, max_value=1.0):
        super().__init__()
        self.max_value = max_value
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    def lpips(self, pred, gt):
        var1 = 2*pred-1
        var2 = 2*gt-1
        LPIPS = self.loss_fn_vgg(var1, var2)
        LPIPS = torch.squeeze(LPIPS)
        
        return LPIPS

    def forward(self, pred, gt):   
        lpips_all = [self.lpips(p.unsqueeze(0), g.unsqueeze(0)) for p, g in zip(pred, gt)]
        lpips = sum(lpips_all) / len(lpips_all)
        return lpips