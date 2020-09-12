# coding: utf-8


import os
import cv2
import shutil
import scipy.io
import numpy as np
# import pandas as pd
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ガウシアン差分フィルタリング
def DoG(img,size, sigma, k=1.6, gamma=1):
    g1 = cv2.GaussianBlur(img, (size, size), sigma)
    g2 = cv2.GaussianBlur(img, (size, size), sigma*k)
    return g1 - gamma*g2

# 閾値で白黒化するDoG
def thres_dog(img, size, sigma, eps, k=1.6, gamma=0.98):
    d = DoG(img,size, sigma, k, gamma)
    d /= d.max()
    d *= 255
    d = np.where(d >= eps, 255, 0)
    return d

# 拡張ガウシアン差分フィルタリング
def xdog(img, size, sigma, eps, phi, k=1.6, gamma=0.98):
    eps /= 255
    d = DoG(img,size, sigma, k, gamma)
    d /= d.max()
    e = 1 + np.tanh(phi*(d-eps))
    e[e>=1] = 1
    return e

# シャープネス値pを使う方
def pxdog(img, size, p, sigma, eps, phi, k=1.6):
    eps /= 255
    g1 = cv2.GaussianBlur(img, (size, size), sigma)
    g2 = cv2.GaussianBlur(img, (size, size), sigma*k)
    d = (1 + p) * g1 - p * g2
    d /= d.max()
    e = 1 + np.tanh(phi*(d-eps))
    e[e>=1] = 1
    return e


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def move_data(data_path, data_list, save_path):

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    for name in tqdm(data_list, ascii=True):
        shutil.copy(os.path.join(data_path, name), os.path.join(save_path, name))
    return None


def make_patch(data_path, save_path, size=256, ch=24, data_key='data'):

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    data_list = os.listdir(data_path)
    for i, name in enumerate(tqdm(data_list, ascii=True)):
        idx = name.split('.')[0]
        # f = scipy.io.loadmat(os.path.join(data_path, name))
        data = Image.open(os.path.join(data_path, name))
        data = np.expand_dims(np.asarray(
            data, np.float32).transpose([2, 0, 1]), axis=0)
        data = normalize(data)
        tensor_data = torch.as_tensor(data)
        patch_data = tensor_data.unfold(2, size, size).unfold(3, size, size)
        patch_data = patch_data.permute(
            (0, 2, 3, 1, 4, 5)).reshape(-1, ch, size, size)
        for i in range(patch_data.size()[0]):
            save_data = patch_data[i].to('cpu').detach().numpy().copy().transpose(1, 2, 0).astype(np.uint8)
            save_name = os.path.join(save_path, f'{idx}_{i}.png')
            save_img = Image.fromarray(save_data)
            save_img.save(save_name)
            # scipy.io.savemat(save_name, {'data': save_data})

    return None


def plot_img(output_imgs, title):
    plt.imshow(output_imgs)
    plt.title('Predict')
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    return None


def psnr(loss):

    return 20 * torch.log10(1 / torch.sqrt(loss))


class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        h, w, _ = img.shape
        i = np.random.randint(0, h - self.size[0], dtype=int)
        j = np.random.randint(0, w - self.size[1], dtype=int)
        return img[i: i + self.size[0], j: j + self.size[1], :].copy()


class RandomHorizontalFlip(object):

    def __init__(self, rate=.5):
        if rate:
            self.rate = rate
        else:
            # self.rate = np.random.randn()
            self.rate = .5

    def __call__(self, img):
        if np.random.randn() < self.rate:
            img = img[:, ::-1, :].copy()
        return img


class ModelCheckPoint(object):

    def __init__(self, checkpoint_path, model_name, mkdir=False, partience=1, verbose=True, *args, **kwargs):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.partience = partience
        self.verbose = verbose
        if mkdir is True:
            if os.path.exists(self.checkpoint_path):
                shutil.rmtree(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.colab2drive_idx = 0
        if 'colab2drive' in kwargs.keys():
            self.colab2drive = kwargs['colab2drive']
            self.colab2drive_path = kwargs['colab2drive_path']
            self.colab2drive_flag = True
        else:
            self.colab2drive_flag = False

    def callback(self, model, epoch, *args, **kwargs):
        if 'loss' not in kwargs.keys() and 'val_loss' not in kwargs.keys():
            assert 'None Loss'
        else:
            loss = kwargs['loss']
            val_loss = kwargs['val_loss']
        checkpoint_name = os.path.join(self.checkpoint_path, self.model_name +
                                       f'_epoch_{epoch:05d}_loss_{loss:.5f}_valloss_{val_loss:.5f}.pth')
        if epoch % self.partience == 0:
            torch.save(model.state_dict(), checkpoint_name)
            if self.verbose is True:
                print(f'CheckPoint Saved by {checkpoint_name}')
        if self.colab2drive_flag is True and epoch == self.colab2drive[self.colab2drive_idx]:
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'optim': kwargs['optim']},
                        os.path.join(self.colab2drive_path, self.model_name + f'_epoch_{epoch:05d}_loss_{loss:.5f}_valloss_{val_loss:.5f}.tar')
                        )
        return self



class Draw_Output(object):

    def __init__(self, img_path, output_data, save_path='output', verbose=False, nrow=8, **kwargs):
        '''
        Parameters
        ---
        img_path: str
          image dataset path
        output_data: list
         draw output data path
        save_path: str(default: 'output')
         output img path
        verbose: bool(default: False)
          verbose
         '''
        self.img_path = img_path
        self.output_data = output_data
        self.data_num = len(output_data)
        self.save_path = save_path
        self.verbose = verbose
        self.shape = kwargs.get('shape', (256, 256))
        self.input_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.shape)),
            torchvision.transforms.ToTensor()
        ])
        self.output_transform = torchvision.transforms.ToPILImage()
        self.nrow = nrow

        ###########################################################
        # Make output directory
        ###########################################################
        if os.path.exists(save_path) is True:
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        if os.path.exists(save_path + '/all_imgs') is True:
            shutil.rmtree(save_path + '/all_imgs')
        os.mkdir(save_path + '/all_imgs')
        ###########################################################
        # Draw Label Img
        ###########################################################
        # labels = []
        # for data in self.output_data:
        #     label = self.input_transform(Image.open(os.path.join(self.img_path, data)).convert('RGB'))
        #     labels.append(label)
        labels = [self.input_transform(Image.open(os.path.join(self.img_path, data)).convert('RGB')) for data in self.output_data]
        self.labels = torch.cat(labels).reshape(len(labels), *labels[0].shape)
        labels_np = torchvision.utils.make_grid(self.labels, nrow=nrow, padding=10)
        labels_np = labels_np.numpy()
        self.labels_np = np.transpose(labels_np, (1, 2, 0))
        del labels, labels_np
        torchvision.utils.save_image(self.labels, os.path.join(save_path, f'labels.jpg'), nrow=nrow, padding=10)


    def callback(self, model, epoch, *args, **kwargs):
        save = kwargs.get('save')
        if 'save' is None:
            assert 'None save mode'
        device = kwargs['device']
        self.epoch_save_path = os.path.join(self.save_path, f'epoch{epoch}')
        os.makedirs(self.epoch_save_path, exist_ok=True)
        output_imgs = []
        # encoder.eval()
        # decoder.eval()
        with torch.no_grad():
            for i, data in enumerate(self.output_data):
                img = self.input_transform(Image.open(os.path.join(self.img_path, data)).convert('L')).unsqueeze(0).to(device)
                output = model(img).squeeze().to('cpu')
                output_imgs.append(output)
        output_imgs = torch.cat(output_imgs).reshape(len(output_imgs), *output_imgs[0].shape)
        if self.verbose is True:
            self.__show_output_img_list(output_imgs)
            # self.__show_output_img_list(self.labels)
        if save is True:
            torchvision.utils.save_image(output_imgs, os.path.join(self.save_path, f'all_imgs/all_imgs_{epoch}.jpg'), nrow=self.nrow, padding=10)
        del output_imgs
        return self

    def __draw_output_label(self, output, label, data):
        output = torch.cat((label, output), dim=2)
        output = self.output_transform(output)
        output.save(os.path.join(self.epoch_save_path, data))
        if self.verbose is True:
            print(f'\rDraw Output {data}', end='')
        return self

    def __show_output_img_list(self, output_imgs):
        plt.figure(figsize=(16, 9))
        output_imgs_np = torchvision.utils.make_grid(output_imgs, nrow=self.nrow, padding=10)
        output_imgs_np = output_imgs_np.numpy()
        output_imgs_np = np.transpose(output_imgs_np, (1, 2, 0))
        plt.subplot(1, 2, 1)
        plot_img(output_imgs_np, 'Predict')
        plt.subplot(1, 2, 2)
        plot_img(self.labels_np, 'Label')
        plt.show()
        del output_imgs_np
        return self
