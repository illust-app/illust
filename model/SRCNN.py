# coding: UTF-8


import numpy as np
import torch
from torchsummary import summary
from .layers import Base_Model, Conv2d_Shuffle


class SRCNN(Base_Model):

    def __init__(self, input_ch, output_ch, kernel_size=3, stride=1, features=[32, 64, 64], **kwargs):
        super(SRCNN, self).__init__()
        # OH = (H + 2P-FH) / S + 1
        # OW = (W + 2P -FW) / S + 1
        self.conv1 = torch.nn.Conv2d(input_ch, 64, kernel_size=9, stride=1, padding=4)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, output_ch, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x


if __name__ == '__main__':

    model = SRCNN(3, 3, activation='swish')
    summary(model, (3, 64, 64))
