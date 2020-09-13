# coding: utf-8


import torch


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


class Swish(torch.nn.Module):

    def forward(self, x):
        return swish(x)


class Mish(torch.nn.Module):

    def forward(self, x):
        return mish(x)


class FReLU(torch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size=3, stride=1):
        super(FReLU, self).__init__()
        self.depth = torch.nn.Conv2d(input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=input_ch)

    def forward(self, x):
        y = self.depth(x)
        return torch.max(x, y)


class Base_Model(torch.nn.Module):

    def __init__(self):
        super(Base_Model, self).__init__()
        self.activation = {None   : torch.nn.Identity(),
                           'relu' : torch.nn.ReLU(),
                           'leaky': torch.nn.LeakyReLU(),
                           'swish': Swish(),
                           'mish' : Mish()}
                           # 'frelu': FReLU(input_ch, output_ch)}


class Conv_Block(torch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, stride, padding, norm=True):
        super(Conv_Block, self).__init__()
        layer = []
        layer.append(torch.nn.Conv2d(input_ch, output_ch,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding))
        if norm is True:
            layer.append(torch.nn.BatchNorm2d(output_ch))
        # layer.append(torch.nn.ReLU())
        self.layer = torch.nn.Sequential(*layer)

    def forward(self, x):
        return torch.nn.functional.relu(self.layer(x))


class D_Conv_Block(torch.nn.Module):

    def __init__(self, input_ch, output_ch, norm=True):
        super(D_Conv_Block, self).__init__()
        layer = [torch.nn.ConvTranspose2d(
            input_ch, output_ch, kernel_size=2, stride=2)]
        if norm is True:
            layer.append(torch.nn.BatchNorm2d(output_ch))
        layer.append(torch.nn.ReLU())
        self.layer = torch.nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class Bottoleneck(torch.nn.Module):

    def __init__(self, input_ch, k):
        super(Bottoleneck, self).__init__()
        bottoleneck = []
        bottoleneck.append(Conv_Block(input_ch, 128, 1, 1, 0))
        bottoleneck.append(Conv_Block(128, k, 3, 1, 1))
        self.bottoleneck = torch.nn.Sequential(*bottoleneck)

    def forward(self, x):
        return self.bottoleneck(x)


class DenseBlock(torch.nn.Module):

    def __init__(self, input_ch, k, layer_num):
        super(DenseBlock, self).__init__()
        bottoleneck = []
        for i in range(layer_num):
            bottoleneck.append(Bottoleneck(input_ch, k))
            input_ch += k
        self.bottoleneck = torch.nn.Sequential(*bottoleneck)

    def forward(self, x):
        for bottoleneck in self.bottoleneck:
            growth = bottoleneck(x)
            x = torch.cat((x, growth), dim=1)
        return x


class TransBlock(torch.nn.Module):

    def __init__(self, input_ch, compress=.5):
        super(TransBlock, self).__init__()
        self.conv1_1 = Conv_Block(input_ch, int(input_ch * compress), 1, 1, 0, norm=False)
        self.ave_pool = torch.nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.ave_pool(x)
        return x


class SA_Block(torch.nn.Module):

    def __init__(self, input_ch):
        super(SA_Block, self).__init__()
        self.theta = torch.nn.Conv2d(input_ch, input_ch // 8, 1, 1, 0)
        self.phi = torch.nn.Conv2d(input_ch, input_ch // 8, 1, 1, 0)
        self.g = torch.nn.Conv2d(input_ch, input_ch, 1, 1, 0)
        # self.attn = torch.nn.Conv2d(input_ch // 2, input_ch, 1, 1, 0)
        self.sigma_ratio = torch.nn.Parameter(
            torch.zeros(1), requires_grad=True)

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        # theta path (first conv block)
        theta = self.theta(x)
        theta = theta.view(batch_size, ch // 8, h *
                           w).permute((0, 2, 1))  # (bs, HW, CH // 8)
        # phi path (second conv block)
        phi = self.phi(x)
        phi = torch.nn.functional.max_pool2d(phi, 2)
        phi = phi.view(batch_size, ch // 8, h * w // 4)  # (bs, CH // 8, HW)
        # attention path (theta and phi)
        attn = torch.bmm(theta, phi)  # (bs, HW, HW // 4)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        # g path (third conv block)
        g = self.g(x)
        g = torch.nn.functional.max_pool2d(g, 2)
        # (bs, HW // 4, CH)
        g = g.view(batch_size, ch, h * w // 4).permute((0, 2, 1))
        # attention map (g and attention path)
        attn_g = torch.bmm(attn, g)  # (bs, HW, CH)
        attn_g = attn_g.permute((0, 2, 1)).view(
            batch_size, ch, h, w)  # (bs, CH, H, W)
        return x + self.sigma_ratio * attn_g


class DW_PT_Conv(torch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, activation='relu'):
        super(DW_PT_Conv, self).__init__()
        self.activation = activation
        self.depth = torch.nn.Conv2d(input_ch, input_ch, kernel_size, 1, 1, groups=input_ch)
        self.point = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)

    def forward(self, x):
        x = self.depth(x)
        x = self._activation_fn(x)
        x = self.point(x)
        x = self._activation_fn(x)
        return x


class EDSR_Block(Base_Model):

    def __init__(self, input_ch, output_ch, feature, **kwargs):
        super(EDSR_Block, self).__init__()
        self.activation_fn = kwargs.get('activation')
        self.conv1 = torch.nn.Conv2d(input_ch, feature, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(feature, output_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_in = x
        x_in = self.activation[self.activation_fn](self.conv1(x_in))
        x_in = self.conv2(x_in)
        return x + x_in

class Conv2d_Shuffle(Base_Model):

    def __init__(self, feature, scale, **kwargs):
        super(Conv2d_Shuffle, self).__init__()
        self.conv = torch.nn.Conv2d(feature, feature * scale ** 2, 3, 1, 1)
        self.pixel_shuffle = torch.nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x
