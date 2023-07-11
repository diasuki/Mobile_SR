import os
import torch
import cv2
import numpy as np
import pickle as pkl
import torchvision.transforms as transforms
from PIL import Image
import random


class SyntheticBurstVal(torch.utils.data.Dataset):
    """ Synthetic burst validation set introduced in [1]. The validation burst have been generated using a
    synthetic data generation pipeline. The dataset can be downloaded from
    https://data.vision.ee.ethz.ch/bhatg/SyntheticBurstVal.zip

    [1] Deep Burst Super-Resolution. Goutam Bhat, Martin Danelljan, Luc Van Gool, and Radu Timofte. CVPR 2021
    """
    def __init__(self, root=None, initialize=True):
        """
        args:
            root - Path to root dataset directory
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        # self.root = os.path.join(root, 'val')
        self.root = root
        self.burst_list = list(range(300))
        self.burst_size = 14

    def initialize(self):
        pass

    def __len__(self):
        return len(self.burst_list)

    def _read_burst_image(self, index, image_id):
        im = cv2.imread('{}/bursts/{:04d}/im_raw_{:02d}.png'.format(self.root, index, image_id), cv2.IMREAD_UNCHANGED)
        im_t = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1).float() / (2**14)

        return im_t

    def _read_gt_image(self, index):
        gt = cv2.imread('{}/gt/{:04d}/im_rgb.png'.format(self.root, index), cv2.IMREAD_UNCHANGED)
        gt_t = (torch.from_numpy(gt.astype(np.float32)) / 2 ** 14).permute(2, 0, 1).float()
        return gt_t

    def _read_meta_info(self, index):
        with open('{}/gt/{:04d}/meta_info.pkl'.format(self.root, index), "rb") as input_file:
            meta_info = pkl.load(input_file)

        return meta_info

    def __getitem__(self, index):
        """ Generates a synthetic burst
        args:
            index: Index of the burst

        returns:
            burst: LR RAW burst, a torch tensor of shape
                   [14, 4, 48, 48]
                   The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
            gt : Ground truth linear image
            meta_info: Meta info about the burst which can be used to convert gt to sRGB space
        """
        burst_name = '{:04d}'.format(index)
        burst = [self._read_burst_image(index, i) for i in range(self.burst_size)]
        burst = torch.stack(burst, 0)

        gt = self._read_gt_image(index)
        meta_info = self._read_meta_info(index)
        meta_info['burst_name'] = burst_name
        return burst, gt, meta_info


class SyntheticBurstValDF2K(torch.utils.data.Dataset):

    def __init__(self, root, burst_size=14, split='val'):
        """
        args:
            root : path of the root directory
            burst_size : Burst size. Maximum allowed burst size is 14.
            crop_sz: Size of the extracted crop. Maximum allowed crop size is 80
            center_crop: Whether to extract a random crop, or a centered crop.
            split: Can be 'train' or 'val'
        """
        assert burst_size <= 14, 'burst_sz must be less than or equal to 14'
        # assert crop_sz <= 80, 'crop_sz must be less than or equal to 80'
        super().__init__()

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.burst_size = burst_size
        self.split = split

        self.root = root
        # split trainset and testset in one dir
        if self.split == 'val':
            root = root + '/test'
        else:
            root = root + '/train'

        self.hrdir = root
        self.lrdir = root
        print(self.lrdir)

        self.lr_filename = '{}/{}/im_rgb_{:02d}'
        self.hr_filename = '{}/{}/im_rgb'

        self.burst_list = self._get_burst_list()
        self.data_length = len(self.burst_list)
        # self.data_length = 20

    def _get_burst_list(self):
        burst_list = sorted(os.listdir(self.lrdir))
        # print(burst_list)
        return burst_list

    def _get_raw_image(self, burst_id, im_id):
        name = self.lr_filename.format(self.lrdir, self.burst_list[burst_id], im_id)

        image = Image.open(f'{name}.png')  # RGB,W, H, C
        image = self.transform(image)
        return image

    def _get_gt_image(self, burst_id):
        name = self.hr_filename.format(self.hrdir, self.burst_list[burst_id])

        image = Image.open(f'{name}.png')  # RGB,W, H, C
        image = self.transform(image)
        return image

    def get_burst(self, burst_id, im_ids):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]
        # pic = self._get_raw_image(burst_id, 0)
        gt = self._get_gt_image(burst_id)
        return frames, gt

    def _sample_images(self):
        burst_size = self.burst_size
        ids = random.sample(range(1, burst_size), k=self.burst_size - 1)
        ids = [0, ] + ids
        return ids

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': 14, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        # Sample the images in the burst, in case a burst_size < 14 is used.
        im_ids = self._sample_images()

        frames, gt = self.get_burst(index, im_ids)
        info = self.get_burst_info(index)

        burst = torch.stack(frames, dim=0)
        burst = burst.float()
        frame_gt = gt.float()

        data = {}
        data['LR'] = burst
        data['HR'] = frame_gt
        data['burst_name'] = info['burst_name']

        return data