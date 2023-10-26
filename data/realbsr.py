from PIL import Image
import os
import random
import json
import os.path
from collections import OrderedDict
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import colour_demosaicing
from libtiff import TIFF
from libtiff import TIFFfile

from data.base import BaseDataset, get_dataloader, stratified_random_split
# from utils.dataset_utils import Augment_RGB_torch

from data.datasets.synthetic_burst_train_set import SyntheticBurstRGBAligned
from data.datasets.synthetic_burst_val_set import SyntheticBurstValDF2K
from data.datasets.zurich_raw2rgb_dataset import DIV2KRGB


class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


def pack_raw_image(im_raw):
    if isinstance(im_raw, np.ndarray):
        im_out = np.zeros_like(im_raw, shape=(4, im_raw.shape[0] // 2, im_raw.shape[1] // 2))
    elif isinstance(im_raw, torch.Tensor):
        im_out = torch.zeros((4, im_raw.shape[0] // 2, im_raw.shape[1] // 2), dtype=im_raw.dtype).to(im_raw.device)
    else:
        raise Exception

    im_out[0, :, :] = im_raw[0::2, 0::2]
    im_out[1, :, :] = im_raw[0::2, 1::2]
    im_out[2, :, :] = im_raw[1::2, 0::2]
    im_out[3, :, :] = im_raw[1::2, 1::2]
    return im_out


def flatten_raw_image(im_raw_4ch):
    """ unpack a 4-channel tensor into a single channel bayer image"""
    if isinstance(im_raw_4ch, np.ndarray):
        im_out = np.zeros_like(im_raw_4ch, shape=(im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2))
    elif isinstance(im_raw_4ch, torch.Tensor):
        im_out = torch.zeros((im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2), dtype=im_raw_4ch.dtype)
    else:
        raise Exception

    im_out[0::2, 0::2] = im_raw_4ch[0, :, :]
    im_out[0::2, 1::2] = im_raw_4ch[1, :, :]
    im_out[1::2, 0::2] = im_raw_4ch[2, :, :]
    im_out[1::2, 1::2] = im_raw_4ch[3, :, :]

    return im_out


augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


def _mod_crop(im, scala):
    w, h = im.size
    return im.crop((0, 0, w - w % scala, h - h % scala))


def get_crop(img, r1, r2, c1, c2):
    im_raw = img[:, r1:r2, c1:c2]
    return im_raw


# manual datasets
class RealBSR(torch.utils.data.Dataset):
    """ Real-world burst super-resolution dataset. """

    def __init__(self, root, crop_sz=64, burst_size=14, split='train'):
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
        assert split in ['train', 'val']
        super().__init__()

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.burst_size = burst_size
        self.crop_sz = crop_sz
        self.split = split

        self.root = root
        # split trainset and testset in one dir
        if self.split == 'val':
            root = root + '/test'
        else:
            root = root + '/train'

        self.hrdir = root # + '/' + 'HR'
        self.lrdir = root # + '/' + 'LR_aligned'
        print(self.lrdir)

        self.lr_filename = '{}/{}/{}_MFSR_Sony_{:04d}_x1_{:02d}'
        self.hr_filename = '{}/{}/{}_MFSR_Sony_{:04d}_x4warp'

        self.substract_black_level = True
        self.white_balance = False

        self.burst_list = self._get_burst_list()
        self.data_length = len(self.burst_list)
        # self.data_length = 20

    def _get_burst_list(self):
        burst_list = sorted(os.listdir(self.lrdir))
        # print(burst_list)
        return burst_list

    def _get_raw_image(self, burst_id, im_id):
        # .../RealBSR/RGB/test/031_0430/031_MFSR_Sony_0430_x1_09.png
        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_number2 = int(self.burst_list[burst_id].split('_')[-1])
        name = self.lr_filename.format(
            self.lrdir, self.burst_list[burst_id], burst_number, burst_number2, im_id)

        image = Image.open(f'{name}.png')  # RGB,W, H, C
        image = self.transform(image)
        return image

    def _get_gt_image(self, burst_id):
        # .../RealBSR/RGB/test/031_0430/031_MFSR_Sony_0430_x4warp.png
        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_nmber2 = int(self.burst_list[burst_id].split('_')[-1])
        name = self.hr_filename.format(
            self.hrdir, self.burst_list[burst_id], burst_number, burst_nmber2)

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

        # Extract crop if needed
        # if frames[0].shape[-1] != self.crop_sz:
        #     print(frames[0].shape, self.crop_sz)
            # r1 = random.randint(0, frames[0].shape[-2] - self.crop_sz)
            # c1 = random.randint(0, frames[0].shape[-1] - self.crop_sz)
            # r2 = r1 + self.crop_sz
            # c2 = c1 + self.crop_sz

            # scale_factor = gt.shape[-1] // frames[0].shape[-1]

            # print(scale_factor)

            # frames = [get_crop(im, r1, r2, c1, c2) for im in frames]
            # gt = get_crop(gt, scale_factor * r1, scale_factor * r2, scale_factor * c1, scale_factor * c2)

        if self.split == 'train':
            apply_trans = transforms_aug[random.getrandbits(3)]
            frames = [getattr(augment, apply_trans)(im) for im in frames]
            gt = getattr(augment, apply_trans)(gt)

        burst = torch.stack(frames, dim=0)
        burst = burst.float()
        frame_gt = gt.float()

        data = {}
        data['LR'] = burst
        data['HR'] = frame_gt
        data['burst_name'] = info['burst_name']

        return data


class RealBSRRAW(RealBSR):

    def __init__(self, root, crop_sz=64, burst_size=14, split='train'):
        super().__init__(root, crop_sz, burst_size, split)
        self.lr_filename = '{}/{}/{}_MFSR_Sony_{:04d}_x1_{:02d}'
        self.hr_filename = '{}/{}/{}_MFSR_Sony_{:04d}_x4_rgb'

    def _get_raw_image(self, burst_id, im_id):
        # .../RealBSR/RAW/test/031_0430/031_MFSR_Sony_0430_x1_09.png
        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_number2 = int(self.burst_list[burst_id].split('_')[-1])
        name = self.lr_filename.format(
            self.lrdir, self.burst_list[burst_id], burst_number, burst_number2, im_id)

        img = cv2.imread(f'{name}.png', cv2.IMREAD_UNCHANGED)  # [H/2, W/2, 4]
        image = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # [4, H/2, W/2]

        # substract black level
        if self.substract_black_level:
            image = image - 512.

        # normalize to [0, 1]
        image = image / 16383.

        return image

    def _get_gt_image(self, burst_id):
        # .../RealBSR/RAW/test/031_0430/031_MFSR_Sony_0430_x4_rgb.png
        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_nmber2 = int(self.burst_list[burst_id].split('_')[-1])
        name = self.hr_filename.format(
            self.hrdir, self.burst_list[burst_id], burst_number, burst_nmber2)
        pkl_path = '{}/{}/{}_MFSR_Sony_{:04d}_x4.pkl'.format(self.hrdir, self.burst_list[burst_id], burst_number, burst_nmber2)

        img = cv2.imread(f'{name}.png', cv2.IMREAD_UNCHANGED)
        image = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # [3, H, W]

        # substract black level
        if self.substract_black_level:
            image = image - 512.

        # normalize to [0, 1]
        image = image / 16383.

        return image, pkl_path

    def get_burst(self, burst_id, im_ids):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]
        # pic = self._get_raw_image(burst_id, 0)
        gt, pkl_path = self._get_gt_image(burst_id)
        return frames, gt, pkl_path

    def __getitem__(self, index):
        # Sample the images in the burst, in case a burst_size < 14 is used.
        im_ids = self._sample_images()

        frames, gt, pkl_path = self.get_burst(index, im_ids)
        info = self.get_burst_info(index)

        # Extract crop if needed
        # if frames[0].shape[-1] != self.crop_sz:
        #     print(frames[0].shape, self.crop_sz)
            # r1 = random.randint(0, frames[0].shape[-2] - self.crop_sz)
            # c1 = random.randint(0, frames[0].shape[-1] - self.crop_sz)
            # r2 = r1 + self.crop_sz
            # c2 = c1 + self.crop_sz

            # scale_factor = gt.shape[-1] // frames[0].shape[-1]

            # print(scale_factor)

            # frames = [get_crop(im, r1, r2, c1, c2) for im in frames]
            # gt = get_crop(gt, scale_factor * r1, scale_factor * r2, scale_factor * c1, scale_factor * c2)

        if self.split == 'train':
            apply_trans = transforms_aug[random.getrandbits(3)]
            frames = [getattr(augment, apply_trans)(im) for im in frames]
            gt = getattr(augment, apply_trans)(gt)

        burst = torch.stack(frames, dim=0)
        burst = burst.float()
        frame_gt = gt.float()

        data = {}
        data['LR'] = burst
        data['HR'] = frame_gt
        data['burst_name'] = info['burst_name']

        # base frame
        flattened_image = flatten_raw_image(data['LR'][0])
        demosaiced_image = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(flattened_image.numpy())
        base_frame = torch.clamp(torch.from_numpy(demosaiced_image).type_as(flattened_image),
                                 min=0.0, max=1.0).permute(2, 0, 1)
        data['base frame'] = base_frame
        data['pkl_path'] = pkl_path

        return data


class RealBSRText(RealBSR):

    def __init__(self, root, crop_sz=64, burst_size=14, split='train'):
        super().__init__(root, crop_sz, burst_size, split)
        self.lr_filename += '_text'
        self.hr_filename += '_text'


class RealBSRTextRAW(RealBSRRAW):

    def __init__(self, root, crop_sz=64, burst_size=14, split='train'):
        super().__init__(root, crop_sz, burst_size, split)
        self.lr_filename += '_text'
        self.hr_filename += '_text'


class RealBSRDataset(BaseDataset):

    def __init__(self, data_dir, space='RGB', size=160, burst_size=14):
        super().__init__(data_dir, size)
        self._RGBDataset = RealBSR
        self._RAWDataset = RealBSRRAW
        self.dir = 'RealBSR'
        self.space = space
        self.burst_size = burst_size

    def get_loader(self, batch_size, num_workers, split='val',
                   world_size=1, rank=0):
        is_train = (split == 'train')
        data_dir = os.path.join(self.data_dir, self.dir, self.space)
        if self.space == 'RGB':
            dataset = self._RGBDataset(data_dir, self.size, self.burst_size, split)
        else:
            dataset = self._RAWDataset(data_dir, self.size, self.burst_size, split)
        kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                      num_replicas=world_size, rank=rank)
        loader = get_dataloader(dataset, shuffle=is_train, drop_last=is_train, **kwargs)
        return loader


class RealBSRTextDataset(RealBSRDataset):

    def __init__(self, data_dir, space='RGB', size=160, burst_size=14):
        super().__init__(data_dir, space, size, burst_size)
        self._RGBDataset = RealBSRText
        self._RAWDataset = RealBSRTextRAW
        self.dir = 'RealBSR_text'


class DF2KDataset(BaseDataset):

    def __init__(self, name, data_dir, space='RGB', size=384, burst_size=14):
        super().__init__(data_dir, size)
        assert space=='RGB', 'RGB Space!'
        self.burst_size = burst_size
        # self.dir = 'SyntheticBurstDF2K'
        self.dir = name

    def get_loader(self, batch_size, num_workers, split='val',
                   world_size=1, rank=0):
        is_train = (split == 'train')
        data_dir = os.path.join(self.data_dir, self.dir)
        if split == 'train':
            dataset = DIV2KRGB(data_dir, split)
            dataset = SyntheticBurstRGBAligned(dataset, burst_size=self.burst_size, crop_sz=self.size)
        elif split == 'val':
            dataset = SyntheticBurstValDF2K(data_dir, burst_size=self.burst_size, split=split)
        else:
            raise Exception

        kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                      num_replicas=world_size, rank=rank)
        loader = get_dataloader(dataset, shuffle=is_train, drop_last=is_train, **kwargs)
        return loader


class DIV2KDataset(BaseDataset):

    def __init__(self, name, data_dir, space='RGB', size=384, burst_size=14):
        super().__init__(data_dir, size)
        assert space=='RGB', 'RGB Space!'
        self.burst_size = burst_size
        # self.dir = 'SyntheticBurstDIV2K'
        self.dir = name

    def get_loader(self, batch_size, num_workers, split='val',
                   world_size=1, rank=0):
        is_train = (split == 'train')
        data_dir = os.path.join(self.data_dir, self.dir)
        if split == 'train':
            dataset = DIV2KRGB(data_dir, split)
            dataset = SyntheticBurstRGBAligned(dataset, burst_size=self.burst_size, crop_sz=self.size)
        elif split == 'val':
            dataset = SyntheticBurstValDF2K(data_dir, burst_size=self.burst_size, split=split)
        else:
            raise Exception

        kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                      num_replicas=world_size, rank=rank)
        loader = get_dataloader(dataset, shuffle=is_train, drop_last=is_train, **kwargs)
        return loader


class QuadRealBSRDataset(BaseDataset):

    def __init__(self, data_dir, space='RGB', size=160, burst_size=14):
        #data_dir:/mnt/SSD2_512GB/szy/dataset/realbsr_quad_align
        super().__init__(data_dir, size)
        self._RGBDataset = RealBSR
        self._RAWDataset = QuadRealBSRRAW
        # self.dir = 'RealBSR'
        self.space = space
        self.burst_size = burst_size

    def get_loader(self, batch_size, num_workers, split='val',
                   world_size=1, rank=0):
        is_train = (split == 'train')
        # data_dir = os.path.join(self.data_dir, self.dir, self.space)
        data_dir = self.data_dir
        if self.space == 'RGB':
            dataset = self._RGBDataset(data_dir, self.size, self.burst_size, split)
        else:
            dataset = self._RAWDataset(data_dir, self.size, self.burst_size, split)
        kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                      num_replicas=world_size, rank=rank)
        loader = get_dataloader(dataset, shuffle=is_train, drop_last=is_train, **kwargs)
        return loader


class QuadRealBSRRAW(RealBSR):

    def __init__(self, root, crop_sz=64, burst_size=14, split='train'):
        super().__init__(root, crop_sz, burst_size, split)
        #        root:/mnt/SSD2_512GB/szy/dataset/realbsr_quad_align
        #   meta_info:root/test/000_0023/000_0023_meta.pkl
        # lr_filename:root/test/000_0023/000_MFSR_Sony_0023_x1_00_quad.tif
        # hr_filename:root/test/000_0023/000_MFSR_Sony_0023_x4warp.png
        self.lr_filename = '{}/{}/{}_MFSR_Sony_{:04d}_x1_{:02d}_quad'
        self.hr_filename = '{}/{}/{}_MFSR_Sony_{:04d}_x4warp'
        # self.hr_filename_quad = '{}/{}/{}_MFSR_Sony_{:04d}_x4warp_quad'
        #burst_list=['000_0023','000_0036'...]
    def _get_raw_image(self, burst_id, im_id):
        # .../RealBSR/RAW/test/031_0430/031_MFSR_Sony_0430_x1_09.png
        # lr_filename:root/test/000_0023/000_MFSR_Sony_0023_x1_00_quad.tif
        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_number2 = int(self.burst_list[burst_id].split('_')[-1])
        name = self.lr_filename.format(
            self.lrdir, self.burst_list[burst_id], burst_number, burst_number2, im_id)
        ############# read lr_tif.tif########################
        lr_tif_path='%s.tif' % (name)
        tif = TIFFfile(lr_tif_path)
        samples, _ = tif.get_samples()
        input_data=samples[0]#[40,40,16]
        image = torch.from_numpy(input_data.astype(np.float32)).float() / (2**14)
        img=image.permute(2,0,1)
        # image = F.interpolate(image.unsqueeze(0), scale_factor=2, mode="bicubic")[0]
        ############# read lr_raw.png########################
        # img = cv2.imread(f'{name}.png', cv2.IMREAD_UNCHANGED)  # [H/2, W/2, 4]
        # image = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # [4, H/2, W/2]
        # # substract black level
        # if self.substract_black_level:
        #     image = image - 512.
        # # normalize to [0, 1]
        # image = image / 16383.

        return img
    # def _get_gray_image(self, burst_id, im_id):
    #     # .../RealBSR/RAW/test/031_0430/031_MFSR_Sony_0430_x1_09.png
    #     # lr_filename:root/test/000_0023/000_MFSR_Sony_0023_x1_00_quad.tif
    #     burst_number = self.burst_list[burst_id].split('_')[0]
    #     burst_number2 = int(self.burst_list[burst_id].split('_')[-1])
    #     name = self.lr_filename.format(
    #         self.lrdir, self.burst_list[burst_id], burst_number, burst_number2, im_id)
    #     ############# read lr_tif.tif########################
    #     lr_tif_path='%s.tif' % (name)
    #     tif = TIFFfile(lr_tif_path)
    #     samples, _ = tif.get_samples()
    #     input_data=samples[0]#[40,40,16]
    #     image = torch.from_numpy(input_data.astype(np.float32)).float() / (2**14)
    #     # image = F.interpolate(image.unsqueeze(0), scale_factor=2, mode="bicubic")[0]
    #     h,w,c=image.shape
    #     num_channels = image.shape[-1] // 4  # 计算输出通道数
    #     output_data = torch.zeros(2*h,2*w,c//4)
    #     # print(output_data.shape)
    #     for i in range(4):
    #         output_data[..., i] = flatten_raw_image(image[..., i * 4 : (i + 1) * 4].permute(2,0,1))
    #     output_data=output_data.permute(2,0,1)
    #     #### rggb->cfa
    #     gray_img=get_gray_image(output_data)
    #     ############# read lr_raw.png########################
    #     # img = cv2.imread(f'{name}.png', cv2.IMREAD_UNCHANGED)  # [H/2, W/2, 4]
    #     # image = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # [4, H/2, W/2]
    #     # # substract black level
    #     # if self.substract_black_level:
    #     #     image = image - 512.
    #     # # normalize to [0, 1]
    #     # image = image / 16383.

    #     return gray_img

    def _get_gt_image(self, burst_id):
        #burst_id 表示文件夹的id
        # .../RealBSR/RAW/test/031_0430/031_MFSR_Sony_0430_x4_rgb.png
        burst_number = self.burst_list[burst_id].split('_')[0]#000
        burst_nmber2 = int(self.burst_list[burst_id].split('_')[-1])#0023
        ##self.hrdir=root/train
        name = self.hr_filename.format(
            self.hrdir, self.burst_list[burst_id], burst_number, burst_nmber2)
        # name_hr_quad = self.hr_filename_quad.format(
        #     self.hrdir, self.burst_list[burst_id], burst_number, burst_nmber2)
        ################## get rgb_hr#################
        image = Image.open(f'{name}.png')  # RGB,W, H, C
        image = self.transform(image)
        
        ################## get raw_hr##################
        # hr_tif_path='%s.tif' % (name_hr_quad)
        # tif = TIFFfile(hr_tif_path)
        # samples, _ = tif.get_samples()
        # input_data=samples[0]#[40,40,16]
        # gt_quad = torch.from_numpy(input_data.astype(np.float32)).float() / (2**14)
        # gt_quad=gt_quad.permute(2,0,1)
        ################## get meta ##################
        
        # meta_info:root/test/000_0023/000_0023_meta.pkl
        pkl_path = '{}/{}/{}_{:04d}_meta.pkl'.format(self.hrdir, self.burst_list[burst_id], burst_number, burst_nmber2)
        # img = cv2.imread(f'{name}.png', cv2.IMREAD_UNCHANGED)#hr_rgb_image
        # image = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # [3, H, W]
        # # substract black level
        # if self.substract_black_level:
        #     image = image - 512.
        # # normalize to [0, 1]
        # image = image / 16383.

        return image, pkl_path
        # return image,gt_quad, pkl_path

    def get_burst(self, burst_id, im_ids):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]
        # grays = [self._get_gray_image(burst_id, i) for i in im_ids]
        # pic = self._get_raw_image(burst_id, 0)
        gt,pkl_path = self._get_gt_image(burst_id)
        return frames, gt, pkl_path

    def __getitem__(self, index):
        # Sample the images in the burst, in case a burst_size < 14 is used.
        im_ids = self._sample_images()

        frames, gt, pkl_path = self.get_burst(index, im_ids)
        info = self.get_burst_info(index)

        if self.split == 'train':
            apply_trans = transforms_aug[random.getrandbits(3)]
            frames = [getattr(augment, apply_trans)(im) for im in frames]
            # grays = [getattr(augment, apply_trans)(im) for im in grays]
            gt = getattr(augment, apply_trans)(gt)
            # gt_quad = getattr(augment, apply_trans)(gt_quad)

        burst = torch.stack(frames, dim=0)
        # grays = torch.stack(grays, dim=0)
        burst = burst.float()
        # grays = grays.float()
        frame_gt = gt.float()
        # frame_gt_quad = gt_quad.float()

        data = {}
        data['LR'] = burst
        # data['LR_grays'] = grays
        data['HR'] = frame_gt
        # data['HR_quad'] = frame_gt_quad
        data['burst_name'] = info['burst_name']

        # base frame
        base_frame=data['LR'][0]
        # flattened_image = flatten_raw_image(data['LR'][0])
        # demosaiced_image = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(flattened_image.numpy())
        # base_frame = torch.clamp(torch.from_numpy(demosaiced_image).type_as(flattened_image),
        #                          min=0.0, max=1.0).permute(2, 0, 1)
        data['base frame'] = base_frame
        data['pkl_path'] = pkl_path

        return data