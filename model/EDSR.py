# coding: utf-8


import numpy as np
import torch
from .layers import EDSR_Block, Conv2d_Shuffle


class EDSR(torch.nn.Module):

    def __init__(self, input_ch, output_ch, block_num=9, feature=64, scale=2, **kwargs):
        super(EDSR, self).__init__()
        activation = kwargs.get('activation')
        self.scale = scale
        self.start_conv = torch.nn.Conv2d(input_ch, feature, 3, 1, 1)
        self.edsr_block = torch.nn.ModuleList([EDSR_Block(input_ch=feature, output_ch=feature, feature=feature * 2, activation=activation) for _ in range(block_num)])
        self.res_conv = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)
        self.factor_conv = torch.nn.Conv2d(output_ch, feature, 1, 1)
        # if scale == 3:
        #     self.up_factor = torch.nn.Sequential(torch.nn.Conv2d(feature, feature * 3 ** 2, 3, 1, 1), torch.nn.PixelShuffle(3))
        # else:
        # self.up_factor = torch.nn.ModuleList([[*torch.nn.Sequential(torch.nn.Conv2d(feature, feature * scale ** 2, 3, 1, 1), torch.nn.PixelShuffle(int(np.log2(scale))))] for _ in range(int(np.log2(scale)))])
        self.up_factor = torch.nn.ModuleList([Conv2d_Shuffle(feature, scale) for _ in range(int(np.log2(scale)))])
        self.output_conv = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)

    def forward(self, x):

        x_in = self.start_conv(x)
        for layer in self.edsr_block:
            x_in = layer(x_in)
        x = self.res_conv(x_in) + x
        x = self.factor_conv(x)
        # x = self.up_factor(x)
        for factor in self.up_factor:
            x = factor(x)
        x = self.output_conv(x)
        return x
