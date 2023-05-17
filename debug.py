import torch
from models import get_model
# from models.unet import UNet
# from data.realbsr import RealBSR

model = get_model('unet', dim=16, burst_size=14, scale=3)
x = torch.randn(2, 14, 3, 16, 16)
y = model(x)
print(x.shape)
print(y.shape)

# ds = RealBSR()

