# coding: UTF-8


import os
import numpy as np
from utils import make_patch, move_data

if __name__ == '__main__':

    data_path = 'save_dataset'
    save_path = 'demo_dataset'
    train_path = os.path.join(save_path, 'train')
    train_patch_path = os.path.join(save_path, 'train_patch')
    test_path = os.path.join(save_path, 'test')
    test_patch_path = os.path.join(save_path, 'test_patch')
    np.random.seed(123456)
    train_test_idx = np.random.choice((0, 1), len(os.listdir(data_path)), p=(.8, .2))
    data_list = os.listdir(data_path)
    data_list.sort()
    data_list = np.array(data_list)
    train_list = list(data_list[train_test_idx == 0])
    test_list = list(data_list[train_test_idx == 1])
    move_data(data_path, train_list, train_path)
    make_patch(train_path, train_patch_path, size=64, ch=3)
    move_data(data_path, test_list, test_path)
    make_patch(test_path, test_patch_path, size=64, ch=3)
