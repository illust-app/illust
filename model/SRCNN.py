# coding: UTF-8


import numpy as np
import torch
from torchsummary import summary
from .layers import Base_Model, Conv2d_Shuffle


class SRCNN(Base_Model):

    def __init__(self, input_ch, output_ch, kernel_size=3, stride=1, features=[32, 64, 64], **kwargs):
        super(SRCNN, self).__init__()
        self.activation_name = kwargs.get('activation')
        self.scale = kwargs.get('scale', 2)
        self.features = [input_ch] + features  # + [output_ch]
        self.layers = torch.nn.ModuleList([torch.nn.Conv2d(self.features[i], self.features[i + 1], kernel_size, stride, padding=kernel_size // 2) for i in range(len(self.features) - 1)])
        self.res_conv = torch.nn.Conv2d(self.features[-1], output_ch, kernel_size, stride, padding=kernel_size // 2)
        self.factor_conv = torch.nn.Conv2d(output_ch, self.features[-1], kernel_size=3, padding=kernel_size // 2)
        self.up_factor = Conv2d_Shuffle(self.features[-1], scale=self.scale)
        self.output_conv = torch.nn.Conv2d(self.features[-1], output_ch, kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, x):
        x_in = x
        for i in range(len(self.layers)):
            x_in = self.layers[i](x_in)
            x_in = self.activation[self.activation_name](x_in)
        x_in = self.res_conv(x_in) + x
        x_in = self.factor_conv(x_in)
        x_in = self.up_factor(x_in)
        x = self.output_conv(x_in)
        return x


if __name__ == '__main__':

    model = SRCNN(3, 3, activation='swish')
    summary(model, (3, 64, 64))
