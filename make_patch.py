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
    draw_path = os.path.join(save_path, 'draw')
    draw_patch_path = os.path.join(save_path, 'draw_patch')
    np.random.seed(123456)
    data_list = os.listdir(data_path)[:5]
    data_list.sort()
    train_test_idx = np.random.choice((0, 1), len(data_list), p=(.8, .2))
    data_list = np.array(data_list)
    train_list = list(data_list[train_test_idx == 0])
    test_list = list(data_list[train_test_idx == 1])
    move_data(data_path, train_list, train_path)
    make_patch(train_path, train_patch_path, size=64, ch=3)
    move_data(data_path, test_list, test_path)
    make_patch(test_path, test_patch_path, size=64, ch=3)
    move_data(data_path, test_list[:max(1, int(len(test_list) * .2))], draw_path)
    # make_patch(draw_path, draw_patch_path, size=256, ch=3)
