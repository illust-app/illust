# coding: utf-8


import numpy as np
# from skimage.measure import compare_mse, compare_psnr, compare_ssim
from PIL import Image
import torch
import torchvision
import pytorch_ssim
import warnings


warnings.simplefilter('ignore')


class RMSEMetrics(torch.nn.Module):

    def __init__(self):
        super(RMSEMetrics, self).__init__()
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.criterion(x, y))


class PSNRMetrics(torch.nn.Module):

    def __init__(self):
        super(PSNRMetrics, self).__init__()
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, y):
        return 10. * torch.log10(1. / self.criterion(x, y))


class SAMMetrics(torch.nn.Module):

    def __init__(self):
        super(SAMMetrics, self).__init__()

    def forward(self, x, y):
        x_sqrt = torch.sqrt(torch.sum(x, dim=0))
        y_sqrt = torch.sqrt(torch.sum(y, dim=0))
        xy = torch.sum(x * y, dim=0)
        angle = torch.acos(xy / (x_sqrt * y_sqrt))
        return torch.mean(angle)


class SSIMLoss(torch.nn.Module):

    def __init__(self, window):
        super(SSIMLoss, self).__init__()
        self.window = window
        sigma = (window - 1) / 2
        x, y = np.arange(window) - sigma, np.arange(window) - sigma
        X, Y = np.meshgrid(x, y)
        self.C1 = .01 ** 2
        self.C2 = .03 ** 2
        self.kernel = torch.Tensor(self._norm2d(x, y, sigma))

    def _norm2d(self, x, y, sigma):
        gauss = -np.exp(-(x ** 2 + y ** 2 / (2 * sigma ** 2))
                        ) / (np.sqrt(2 * np.pi) * sigma)
        return gauss / gauss.sum()

    def forward(self, x, y):
        channel = x.size()[0]
        norm_x = torch.nn.functional.conv2d(
            x, self.kernel, padding=self.window // 2, groups=channel)
        norm_y = torch.nn.functional.conv2d(
            y, self.kernel, padding=self.window // 2, groups=channel)

        norm_x2 = norm_x ** 2
        norm_y2 = norm_y ** 2
        norm_xy = norm_x * norm_y
        sigma_x = torch.nn.functional.conv2d(
            x * x, self.kernel, padding=self.window // 2, groups=channel) - norm_x2
        sigma_y = torch.nn.functional.conv2d(
            y * y, self.kernel, padding=self.window // 2, groups=channel) - norm_y2
        sigma_xy = torch.nn.functional.conv2d(
            x * y, self.kernel, padding=self.window // 2, groups=channel) - norm_xy

        ssim = ((2 * norm_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
            ((norm_x2 + norm_y2 + self.C1) * (sigma_x + sigma_y + self.C2))

        return torch.mean(ssim)


if __name__ == '__main__':

    img_x = Image.open('Lenna.bmp')
    nd_x = np.asarray(img_x, dtype=np.float32) / 255.
    x = torchvision.transforms.ToTensor()(img_x).unsqueeze(0)
    # img_y = Image.open('Lenna_000.jpg')
    img_y = Image.open('Lenna.bmp')
    nd_y = np.asarray(img_y, dtype=np.float32) / 255.
    y = torchvision.transforms.ToTensor()(img_y).unsqueeze(0)
    mse = torch.nn.MSELoss()
    rmse = RMSEMetrics()
    psnr = PSNRMetrics()
    ssim = SSIMLoss(window=11)
    print(x.max())
    print(x.min())
    print(y.max())
    print(y.min())

    print('mine:', mse(x, y))
    print('mine:', rmse(x, y))
    print('mine:', psnr(x, y))
    print('mine:', ssim(x, y))

    print(nd_x.shape, nd_y.shape)

    print('skimage:', compare_mse(nd_x, nd_y))
    print('skimage:', np.sqrt(compare_mse(nd_x, nd_y)))
    print('skimage:', compare_psnr(nd_x, nd_y))
    print('skimage:', compare_ssim(nd_x, nd_y, multichannel=True))

    # print('mse :', mse_evaluate(x, y))
    # print('rmse:', rmse_evaluate(x, y))
    # print('ssim:', ssim_evaluate(x, y))

    print('ssim:', pytorch_ssim.ssim(x, y))
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    print('ssim:', ssim_loss(x, y))
