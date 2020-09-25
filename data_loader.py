# coding


import os
import torch
import torchvision
import numpy as np
# import scipy.io as sio
from PIL import Image
# from utils import normalize


data_path = 'dataset/'
size = 256


class SRDataset(torch.utils.data.Dataset):

    def __init__(self, img_path, scale=2, transform=None):

        self.img_path = img_path
        self.data = os.listdir(img_path)
        self.data_len = len(self.data)
        self.scale = scale
        self.to_tensor = torchvision.transforms.ToTensor()
        self.transforms = transform

    def __getitem__(self, idx):
        # mat_data = sio.loadmat(os.path.join(self.img_path, self.data[idx]))['data']
        # mat_data = mat_data['data']
        img_name = os.path.join(self.img_path, self.data[idx])
        img = Image.open(img_name).convert('RGB')
        nd_data = np.array(img, dtype=np.float32).copy()
        h, w, ch = nd_data.shape
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = self.to_tensor(nd_data)
        # label_img = self.to_tensor(nd_data)
        label_data = nd_data
        img = nd_data.detach().numpy().astype(np.uint8).transpose(1, 2, 0)
        img = Image.fromarray(img)
        input_data = img.resize((h // self.scale, w // self.scale))
        input_data = self.to_tensor(np.array(input_data, dtype=np.float32))
        # input_data, label_data = 2 * normalize(input_data) - 1, 2 * normalize(label_data) - 1
        return input_data, label_data

    def __len__(self):
        return self.data_len


class SRCNNDataset(SRDataset):

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path, self.data[idx])
        img = Image.open(img_name).convert('RGB')
        nd_data = np.array(img, dtype=np.float32).copy()
        h, w, ch = nd_data.shape
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = self.to_tensor(nd_data)
        label_data = nd_data
        '''
        img = nd_data.detach().numpy().astype(np.uint8).transpose(1, 2, 0)
        img = Image.fromarray(img)
        input_data = img.resize((h // self.scale, w // self.scale))
        input_data = img.resize((h * self.scale, w * self.scale))
        '''
        input_data = nd_data.unsqueeze(0)
        input_data = torch.nn.functional.adaptive_avg_pool2d(input_data, (h // self.scale, w // self.scale))
        input_data = torch.nn.functional.interpolate(input_data, (h, w), mode='bicubic', align_corners=False).squeeze()
        return input_data, label_data
