import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata
import glob
import os
import hdf5storage as hdf5
import scipy.io as scio
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

valid_name_list = ['hairs_ms', 'balloons_ms', 'real_and_fake_peppers_ms', 'stuffed_toys_ms', 'thread_spools_ms', 'fake_and_real_tomatoes_ms', 'fake_and_real_lemons_ms', 'egyptian_statue_ms', 'clay_ms', 'real_and_fake_apples_ms', 'fake_and_real_beers_ms', 'fake_and_real_peppers_ms']
train_name_list = ['watercolors_ms', 'beads_ms', 'fake_and_real_sushi_ms', 'pompoms_ms', 'sponges_ms', 'cloth_ms', 'oil_painting_ms', 'flowers_ms', 'cd_ms', 'superballs_ms', 'fake_and_real_lemon_slices_ms', 'fake_and_real_food_ms', 'paints_ms', 'face_ms', 'feathers_ms', 'chart_and_stuffed_toy_ms', 'jelly_beans_ms', 'photo_and_face_ms', 'fake_and_real_strawberries_ms', 'glass_tiles_ms']


class HyperDatasetValid(udata.Dataset):
    def __init__(self, mode='valid'):
        # if mode != 'valid':
        #     raise Exception("Invalid mode!", mode)
        data_path = '/home/data/dusongcheng/dataset/CAVE/CAVE_Validation_Spectral'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))
        self.res_path = '/home/data/dusongcheng/dataset/resp.mat'
        self.keys = data_names
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # mat = h5py.File(self.keys[index], 'r')
        mat = hdf5.loadmat(self.keys[index])
        res = hdf5.loadmat(self.res_path)['resp']
        res = np.transpose(res, (1, 0))
        mat_name = self.keys[index].split('/')[-1]
        hyper = np.float32(np.array(mat['rad'])/(2**16-1))
        hyper1 = cv2.GaussianBlur(hyper,(7,7),2)[3::8,3::8,:]
        rgb = np.tensordot(hyper, res, (-1, 0))
        hyper1 = np.transpose(hyper1, [2, 0, 1])
        hyper = np.transpose(hyper, [2, 0, 1])
        rgb = np.transpose(rgb, [2, 0, 1])
        hyper = torch.Tensor(hyper)
        hyper1 = torch.Tensor(hyper1)
        rgb = torch.Tensor(rgb)
        return rgb, hyper, hyper1, mat_name

class HyperDatasetTrain1(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        data_names = train_name_list
        self.baseroot = './dataset/CAVE/CAVE_Train_Spectral'
        self.keys = data_names
        self.res_path = './dataset/resp.mat'
        self.num_pre_img = 4
        self.train_len = 20*16
        self.test_len = 12
    def __len__(self):
        return len(self.keys)*16

    def __getitem__(self, index):
        index_img = index // self.num_pre_img**2 
        index_inside_image = index % self.num_pre_img**2 
        index_row = index_inside_image // self.num_pre_img 
        index_col = index_inside_image % self.num_pre_img

        mat_path = os.path.join(self.baseroot, self.keys[index_img])
        mat = hdf5.loadmat(mat_path)
        res = hdf5.loadmat(self.res_path)['resp']
        res = np.transpose(res, (1, 0))

        hyper = np.float32(np.array(mat['rad']/(2**16-1)))
        
        temp_a = cv2.GaussianBlur(hyper,(7,7),2)[3::8,3::8,:]
        hsi_g = hyper[index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
        hsi = temp_a[index_row*16:(index_row+1)*16,index_col*16:(index_col+1)*16,:]
        msi = np.tensordot(hsi_g, res,(-1,0))
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        for j in range(rotTimes):
            hsi_g = np.rot90(hsi_g)
            hsi = np.rot90(hsi)
            msi = np.rot90(msi)

        # Random vertical Flip   
        for j in range(vFlip):
            hsi_g = np.flip(hsi_g,axis=1)
            hsi = np.flip(hsi,axis=1)
            msi = np.flip(msi,axis=1)
    
        # Random Horizontal Flip
        for j in range(hFlip):
            hsi_g = np.flip(hsi_g,axis=0)
            hsi = np.flip(hsi,axis=0)
            msi = np.flip(msi,axis=0)

        hsi = np.transpose(hsi,(2,0,1)).copy()
        msi = np.transpose(msi,(2,0,1)).copy()
        hsi_g = np.transpose(hsi_g,(2,0,1)).copy()

        hsi = torch.Tensor(hsi)
        msi = torch.Tensor(msi)
        hsi_g = torch.Tensor(hsi_g)

        return msi, hsi_g, hsi

