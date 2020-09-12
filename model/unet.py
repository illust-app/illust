# coding: utf-8


import torch
import torchvision
from .base_model import CNN_Block, CNN_Block_for_UNet


class UNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch):
        super(UNet, self).__init__()
        #############################################################################################################
        # Encoder Block
        #############################################################################################################
        out_feature_list = [input_ch] + [64, 128, 256, 512, 1024]
        encode_feature = out_feature_list
        encode_block = []
        encode_block.append(CNN_Block_for_UNet(
            encode_feature[0], encode_feature[1], kernel=3, pool=False, num_layer=2))
        for i in range(1, len(encode_feature) - 1):
            encode_block.append(CNN_Block_for_UNet(
                encode_feature[i], encode_feature[i + 1], kernel=3, num_layer=2))
        self.encode_block = torch.nn.Sequential(*encode_block)
        #############################################################################################################
        # Decoder Block
        #############################################################################################################
        decode_feature = encode_feature[1:][::-1]
        up_block = []
        decode_block = []
        for i in range(len(decode_feature) - 1):
            up_block.append(torch.nn.ConvTranspose2d(
                decode_feature[i], decode_feature[i + 1], kernel_size=2, stride=2))
            decode_block.append(CNN_Block(
                decode_feature[i + 1] * 2, decode_feature[i + 1], kernel=3, num_layer=2, pool=False))
        self.up_block = torch.nn.Sequential(*up_block)
        self.decode_block = torch.nn.Sequential(*decode_block)
        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(decode_feature[-1],
                            output_ch, kernel_size=1, padding=0),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoder = []
        for encode in self.encode_block:
            x = encode(x)
            encoder.append(x)
        encoder = encoder[::-1]
        x_up = encoder[0]
        for i in range(len(self.decode_block)):
            x_up = self.up_block[i](x_up)
            x_up = torch.cat([x_up, encoder[i + 1]], dim=1)
            x_up = self.decode_block[i](x_up)
        output = self.output(x_up)
        return output
