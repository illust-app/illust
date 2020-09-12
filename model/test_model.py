# coding: utf-8


import torch
import layers


class Test_Model(torch.nn.Module):

    def __init__(self, input_ch, output_ch):
        super(Test_Model, self).__init__()

        feature_list = [32, 64, 128, 256, 512]
        pool_flag = [True for _ in range(len(feature_list))]
        pool_flag[-1] = False
        self.start_conv = torch.nn.Conv2d(input_ch, feature_list[0], 3, 1, 1)
        self.start_batch = torch.nn.BatchNorm2d(feature_list[0])
        encode_layer = []
        for i in range(len(feature_list) - 1):
            encode_layer.append(layers.DW_PT_Conv(feature_list[i], feature_list[i + 1], 3, 'swish'))
