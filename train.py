# coding: UTF-8


import os
import torch
from torchsummary import summary
from data_loader import SRDataset
from model.SRCNN import SRCNN
from model.EDSR import EDSR
from trainer import Trainer
from utils import Draw_Output, ModelCheckPoint


batch_size = 64
epochs = 5
scale = 8


device = 'cpu'


data_path = 'demo_dataset'
train_data_path = os.path.join(data_path, 'train_patch')
test_data_path = os.path.join(data_path, 'test_patch')


if __name__ == '__main__':

    train_dataset = SRDataset(train_data_path, scale=scale)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = SRDataset(test_data_path, scale=scale)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = SRCNN(3, 3, features=[32, 64, 128, 256, 128, 64, 32], activation='frelu', scale=scale).to(device)
    # model = EDSR(3, 3, block_num=6, activation='relu', scale=2).to(device)
    summary(model, (3, 64 // scale, 64 // scale))
    criterion = torch.nn.MSELoss().to(device)
    params = list(model.parameters())
    optim = torch.optim.Adam(params=params, lr=1e-3)
    ckpt_callbacks = ModelCheckPoint('ckpt', 'sr', mkdir=True, partience=1, verbose=True)
    draw_callbacks = Draw_Output(os.path.join(data_path, 'draw'), 'show_output', partience=2, shape=(256, 256), scale=scale)
    callbacks = [ckpt_callbacks, draw_callbacks]
    trainer = Trainer(model, criterion, optim, shape=(batch_size, 3, 64, 64), colab_flas=False, callbacks=callbacks)
    trainer.train(epochs, train_dataloader, test_dataloader)
