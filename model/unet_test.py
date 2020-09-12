# coding: utf-8


import torch
from base_model import Conv_Block_UNet


class UNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch):
        super(UNet, self).__init__()

        feature_list = [32, 64, 128, 256, 512, 1024]

        self.encode_1_1 = Conv_Block_UNet(input_ch, feature_list[0], 3, 1, 1)
        self.encode_1_2 = Conv_Block_UNet(
            feature_list[0], feature_list[0], 3, 1, 1)
        self.encode_1_3 = Conv_Block_UNet(
            feature_list[0], feature_list[0], 3, 1, 1)
        self.pool1 = torch.nn.Ma2Pool2d(2)
        self.encode_2_1 = Conv_Block_UNet(
            feature_list[0], feature_list[1], 3, 1, 1)
        self.encode_2_2 = Conv_Block_UNet(
            feature_list[1], feature_list[1], 3, 1, 1)
        self.encode_2_3 = Conv_Block_UNet(
            feature_list[1], feature_list[1], 3, 1, 1)
        self.pool2 = torch.nn.Ma2Pool2d(2)
        self.encode_3_1 = Conv_Block_UNet(
            feature_list[1], feature_list[2], 3, 1, 1)
        self.encode_3_2 = Conv_Block_UNet(
            feature_list[2], feature_list[2], 3, 1, 1)
        self.encode_3_3 = Conv_Block_UNet(
            feature_list[2], feature_list[2], 3, 1, 1)
        self.pool3 = torch.nn.Ma2Pool2d(2)
        self.encode_4_1 = Conv_Block_UNet(
            feature_list[2], feature_list[3], 3, 1, 1)
        self.encode_4_2 = Conv_Block_UNet(
            feature_list[3], feature_list[3], 3, 1, 1)
        self.pool4 = torch.nn.Ma2Pool2d(2)
        self.encode_5_1 = Conv_Block_UNet(
            feature_list[3], feature_list[4], 3, 1, 1)
        self.encode_5_2 = Conv_Block_UNet(
            feature_list[4], feature_list[4], 3, 1, 1)
        self.pool5 = torch.nn.Ma2Pool2d(2)
        self.encode_6_1 = Conv_Block_UNet(
            feature_list[4], feature_list[5], 3, 1, 1)
        self.encode_6_2 = Conv_Block_UNet(
            feature_list[5], feature_list[5], 3, 1, 1)

        self.deconv5 = torch.nn.ConvTranspose2d(
            feature_list[5], feature_list[4], 2, 2)
        self.decode_5_1 = Conv_Block_UNet(
            feature_list[4] * 2, feature_list[4], 3, 1, 1)
        self.decode_5_2 = Conv_Block_UNet(
            feature_list[4], feature_list[4], 3, 1, 1)
        self.deconv4 = torch.nn.ConvTranspose2d(
            feature_list[4], feature_list[3], 2, 2)
        self.decode_4_1 = Conv_Block_UNet(
            feature_list[3] * 2, feature_list[3], 3, 1, 1)
        self.decode_4_2 = Conv_Block_UNet(
            feature_list[3], feature_list[3], 3, 1, 1)
        self.deconv3 = torch.nn.ConvTranspose2d(
            feature_list[3], feature_list[2], 2, 2)
        self.decode_3_1 = Conv_Block_UNet(
            feature_list[2] * 2, feature_list[2], 3, 1, 1)
        self.decode_3_2 = Conv_Block_UNet(
            feature_list[2], feature_list[2], 3, 1, 1)
        self.deconv2 = torch.nn.ConvTranspose2d(
            feature_list[2], feature_list[1], 2, 2)
        self.decode_2_1 = Conv_Block_UNet(
            feature_list[1] * 2, feature_list[1], 3, 1, 1)
        self.decode_2_2 = Conv_Block_UNet(
            feature_list[1], feature_list[1], 3, 1, 1)
        self.deconv1 = torch.nn.ConvTranspose2d(
            feature_list[1], feature_list[0], 2, 2)
        self.decode_1_1 = Conv_Block_UNet(
            feature_list[0] * 2, feature_list[0], 3, 1, 1)
        self.decode_1_2 = Conv_Block_UNet(
            feature_list[0], feature_list[0], 3, 1, 1)

        self.output = torch.nn.Conv2d(feature_list[0], output_ch, 1, 1)

    def forward(self, x):
