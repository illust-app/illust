# coding: UTF-8


import os
import torch
from torchsummary import summary
from data_loader import SRDataset
from model.SRCNN import SRCNN
from trainer import Trainer


device = 'cpu'


data_path = 'demo_dataset'
train_data_path = os.path.join(data_path, 'train_patch')
test_data_path = os.path.join(data_path, 'test_patch')


if __name__ == '__main__':

    train_dataset = SRDataset(train_data_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_dataset = SRDataset(test_data_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=True)
    model = SRCNN(3, 3, features=[32, 64, 128, 256, 128, 64, 32], activation='frelu').to(device)
    # summary(model, (3, 64, 64))
    criterion = torch.nn.MSELoss().to(device)
    params = list(model.parameters())
    optim = torch.optim.Adam(params=params, lr=1e-3)
    trainer = Trainer(model, criterion, optim, shape=(5, 3, 64, 64))
    trainer.train(5, train_dataloader, test_dataloader)
