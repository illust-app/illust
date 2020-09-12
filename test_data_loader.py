# coding: UTF-8


from data_loader import SRDataset
import torch


if __name__ == '__main__':

    data_path = 'patch_dataset'
    datasets = SRDataset(data_path)
    for i, (x, y) in enumerate(datasets):
        print(i, x.shape, y.shape)
    data_loader = torch.utils.data.DataLoader(datasets, batch_size=2, shuffle=True)
    for i, (x, y) in enumerate(data_loader):
        print(i, x.shape, y.shape)
