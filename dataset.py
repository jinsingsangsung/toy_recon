import os
import torch
from torch.utils.data import Dataset
from math import pi, sqrt, ceil
import torch.nn.functional as F
import numpy as np
from matplotlib.path import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms.functional import center_crop, resize
from torchvision.io import read_image
from torch.nn.functional import interpolate
import decord
from einops import rearrange
import imageio


class VideoDataSet(Dataset):
    def __init__(self, path, laplacian=False):
        if os.path.isfile(path):
            self.video = decord.VideoReader(path)
        else:
            self.video = [os.path.join(path, x) for x in sorted(os.listdir(path))]

        first_frame = self.img_load(0)
        self.h, self.w = first_frame.size(-2), first_frame.size(-1)
        self.final_size = self.h * self.w
        self.laplacian = laplacian

    def img_load(self, idx):
        if isinstance(self.video, list):
            img = read_image(self.video[idx])
        else:
            img = self.video[idx].permute(-1,0,1)
        return img / 255.


    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        tensor_image = self.img_load(idx)
        norm_idx = float(idx) / len(self.video)
        sample = {'idx': idx, 'norm_idx': norm_idx}
        
        # Progressive average pooling
        H, W = tensor_image.shape[-2:]

        img_1 = F.avg_pool2d(tensor_image, kernel_size=5, stride=5)
        img_2 = F.avg_pool2d(img_1, kernel_size=4, stride=4)
        img_3 = F.avg_pool2d(img_2, kernel_size=4, stride=4)
        img_4 = F.avg_pool2d(img_3, kernel_size=2, stride=2)
        img_5 = F.avg_pool2d(img_4, kernel_size=2, stride=2)
        
        if self.laplacian:
            # Upsample img_1 back to original size
            img_1_up = F.interpolate(img_1[None], size=tensor_image.shape[-2:], mode='bilinear', align_corners=False)[0]
            img_2_up = F.interpolate(img_2[None], size=img_1.shape[-2:], mode='bilinear', align_corners=False)[0]  
            img_3_up = F.interpolate(img_3[None], size=img_2.shape[-2:], mode='bilinear', align_corners=False)[0]
            img_4_up = F.interpolate(img_4[None], size=img_3.shape[-2:], mode='bilinear', align_corners=False)[0]
            img_5_up = F.interpolate(img_5[None], size=img_4.shape[-2:], mode='bilinear', align_corners=False)[0]
            
            # Compute Laplacian pyramid level by subtracting upsampled lower resolution
            sample['img'] = tensor_image - img_1_up
            sample['img_1'] = img_1 - img_2_up
            sample['img_2'] = img_2 - img_3_up
            sample['img_3'] = img_3 - img_4_up
            sample['img_4'] = img_4 - img_5_up
            sample['img_5'] = img_5
        else:
            sample['img'] = tensor_image
            sample['img_1'] = img_1
            sample['img_2'] = img_2
            sample['img_3'] = img_3
            sample['img_4'] = img_4
            sample['img_5'] = img_5
        
        # # _, _, H, W = tensor_image.shape
        # img_1 = tensor_image[:, :(H//5)*5-2:5, :(W//5)*5-2:5]  # Sample every 5th pixel while ensuring output size is H/5, W/5
        # sample['img_1'] = img_1
        
        # # Pool with kernel size 4x4 and stride 4 on img_1 
        # img_2 = img_1[:, :(H//4)*4-2:4, :(W//4)*4-2:4]
        # sample['img_2'] = img_2
        
        # # Pool with kernel size 4x4 and stride 4 on img_2
        # img_3 = img_2[:, :(H//4)*4-2:4, :(W//4)*4-2:4]
        # sample['img_3'] = img_3
        
        # # Pool with kernel size 2x2 and stride 2 on img_3
        # img_4 = img_3[:, :(H//2)*2-1:2, :(W//2)*2-1:2]
        # sample['img_4'] = img_4
        
        # # Pool with kernel size 2x2 and stride 2 on img_4
        # img_5 = img_4[:, :(H//2)*2-1:2, :(W//2)*2-1:2]        
        # sample['img_5'] = img_5

        return sample