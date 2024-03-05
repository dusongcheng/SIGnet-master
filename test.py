import argparse
import os
import torch
import numpy as np
import cv2
import hdf5storage as hdf5
import time
import utils1 as utils
from dataset import HyperDatasetValid
import skimage.measure as skm

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parse = argparse.ArgumentParser()
    parse.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parse.add_argument('--activ', type = str, default = 'lrelu', help = 'activation type of networks')
    parse.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parse.add_argument('--in_channels', type = int, default = 31, help = 'input channels for generator')
    parse.add_argument('--out_channels', type = int, default = 31, help = 'output channels for generator')
    parse.add_argument('--start_channels', type = int, default = 48, help = 'start channels for generator')
    opt = parse.parse_args()
    name = './model/CAVE_8.pkl'
    # Initialize
    model = utils.create_generator_val(opt, name).cuda()
    test_dataset = HyperDatasetValid(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = True, pin_memory = True)
    print('Network name: %s;' %name)
    criterion_valid = utils.Loss_valid().cuda()
    loss = utils.validate(test_loader, model, criterion_valid)
    print(loss)
    print('ssim, rmse, psnr, ergas, sam()')