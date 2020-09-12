# coding: utf-8


import os
from tqdm import tqdm
import shutil
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from data_loader import RefineEvaluateDataset
from model.unet import UNet
from evaluate import RMSEMetrics, PSNRMetrics
from pytorch_ssim import SSIM
from utils import RefineEvaluater


data_path = 'dataset'
test_path = os.path.join(data_path, 'test')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':

    dataset = RefineEvaluateDataset('output_deeper_mat', test_path, band=10)
    print('load dataset')
    model = UNet(1, 1).to(device)
    model.load_state_dict(torch.load(
        'trained_refine.pth', map_location=torch.device('cpu')))
    print('load model')
    rmse_evaluate = RMSEMetrics().to(device)
    psnr_evaluate = PSNRMetrics().to(device)
    ssim_evaluate = SSIM().to(device)
    evaluate_fn = [rmse_evaluate, psnr_evaluate, ssim_evaluate]
    print('load evaluate_fn')
    evaluater = RefineEvaluater(10, 'output_deeper_refine', 'output_deeper_refine_mat',
    'output_deeper_refine.csv')
    print('load evaluater')
    evaluater.metrics(model, dataset, evaluate_fn,
                      header=['RMSE', 'PSNR', 'SSIM'])
    print('done')