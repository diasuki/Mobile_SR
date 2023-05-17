import os
import numpy as np
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from data.base import get_dataloader

class ImageToImage(VisionDataset):

    def __init__(self, root: str, input_dir: str, target_dir: str,
                 transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)
    
    @staticmethod
    def get_samples(root: str, input_dir: str, target_dir: str):
        input_dir = os.path.join(root, input_dir)
        target_dir = os.path.join(root, target_dir)
