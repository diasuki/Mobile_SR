import torch
from models import get_model
from thop import profile
from thop import clever_format
# from models.unet import UNet
# from data.realbsr import RealBSR

# model = get_model('unet_pares4_new', dim=64, burst_size=14, in_channel=3, scale=4)
# model = get_model('uformer', dim=64, burst_size=14, in_channel=3, scale=4)
# model = get_model('uformer_leff', dim=64, burst_size=14, in_channel=3, scale=4)
# model = get_model('uformer_anti', dim=64, burst_size=14, in_channel=3, scale=4)
# model = get_model('uformer_tiny', dim=64, burst_size=14, in_channel=3, scale=4)
# model = get_model('uformer_tiny_leff', dim=64, burst_size=14, in_channel=3, scale=4)
# model = get_model('uformer_tiny_anti_7', dim=64, burst_size=14, in_channel=3, scale=4)
# model = get_model('safmn', dim=64, burst_size=14, in_channel=3, scale=4)
# model = get_model('shufflemixer', dim=64, burst_size=14, in_channel=3, scale=4)
model = get_model('swinir', dim=64, burst_size=14, in_channel=3, scale=4)
print(model)
x = torch.randn(1, 14, 3, 160, 160)
# y = model(x)
# print(x.shape)
# print(y.shape)

# ds = RealBSR()

# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"params: {params/10**6}M")
macs, params = profile(model, inputs=(x, ))
macs, params = clever_format([macs, params], "%.3f")
print((macs, params))