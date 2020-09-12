# coding: utf-8


import torch
import torchvision
from torchsummary import summary
from .base_model import CNN_Block, D_CNN_Block


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(torch.nn.Module):
    
    def __init__(self, out_feature_list=[32, 64, 128], num_layer=3):
        super(Encoder, self).__init__()
        block = []
        feature_list = [1] + out_feature_list
        for i in range(len(feature_list) - 2):
            block.append(CNN_Block(feature_list[i], feature_list[i + 1], kernel=3, num_layer=num_layer))
        block.append(CNN_Block(feature_list[-2], feature_list[-1], kernel=3, pool=False))
        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

    
class Decoder(torch.nn.Module):
    
    def __init__(self, out_feature_list=[128, 64, 32], num_layer=3):
        super(Decoder, self).__init__()
        block = []
        feature_list = out_feature_list
        for i in range(len(feature_list) - 1):
            block.append(D_CNN_Block(feature_list[i], feature_list[i + 1], kernel=3, num_layer=num_layer))
        self.block = torch.nn.Sequential(*block)
        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(feature_list[-1], 3, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block(x)
        x = self.output(x)
        return x


class CAE(torch.nn.Module):

    def __init__(self, out_feature_list, num_layer):
        super(CAE, self).__init__()

        encoder_feature = out_feature_list
        self.encoder = Encoder(encoder_feature, num_layer)
        decoder_feature = out_feature_list[::-1]
        self.decoder = Decoder(decoder_feature, num_layer)

    def forward(self, x):

        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


if __name__ == '__main__':

    out_feature_list = [32, 64, 128, 256, 512]
    model = CAE(out_feature_list).to(device)
    print(model)
    summary(model, (1, 96, 96))
    torch.save(model.encoder.state_dict(), 'encoder.pth')
    torch.save(model.decoder.state_dict(), 'decoder.pth')
    torch.save(model.state_dict(), 'model.pth')
