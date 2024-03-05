import os
import math
from cv2 import recoverPose
import numpy as np
import torch
import torch.nn as nn
import logging
import torchvision as tv
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
import hdf5storage as hdf5
from network_code1 import SGN
import time
import skimage.measure as skm
# ----------------------------------------
#                 Network
# ----------------------------------------


def create_generator_val(opt, load_name):
    # Initialize the network
    generator = SGN(opt)
    save_point = torch.load(load_name)
    model_param = save_point['spenet']
    model_dict = {}
    for k1, k2 in zip(generator.state_dict(), model_param):
        model_dict[k1] = model_param[k2]
    generator.load_state_dict(model_dict)
    return generator

def load_model(model,mode_dict):
    # state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in mode_dict.items():
        # name = k[7:] # remove `module.`
        name = k # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

# Validate
def validate(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter_valid()
    for i, (img_B, img_A, img_A_1, img_name) in enumerate(val_loader):
        # To device
        img_A = img_A.cuda()
        img_A_1 = img_A_1.cuda()
        img_B = img_B.cuda()

        # Forward propagation
        with torch.no_grad():
            start_time = time.time()
            out = model(img_B, img_A_1)  # [0:480, 0:512, :], [1, 31, 480, 512]
            # print(time.time()-start_time)
            
        loss = criterion(img_A, out[-1])
        # print(img_A.max(), out[-1].max())
        # max_loss = torch.abs(img_A-out).max()
        # out = out.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)
        # save_img_path = os.path.join('/home/data/dusongcheng/CSPN/CSPN-mywork/CAVE_8/results', img_name[0])
        # print(save_img_path, max_loss)
        # hdf5.write(data=out, path='cube', filename=save_img_path, matlab_compatible=True)

        # record loss
        losses.update(loss.data)

    return losses.avg

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeter_valid(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = np.zeros([1,5])
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*np.array(n)
        self.count += n
        self.avg = self.sum / self.count

class Loss_valid(nn.Module):
    def __init__(self):
        super(Loss_valid, self).__init__()

    def forward(self, label_image, rec_image):
        self.batch_size = label_image.shape[0]
        assert self.batch_size == 1
        self.label = label_image.data.cpu().squeeze(0).numpy()
        self.output = rec_image.data.cpu().squeeze(0).numpy()
        self.output = np.clip(self.output, 0, 1)
        valid_error = np.zeros([1, 5])

        valid_error[0, 0] = self.ssim()
        valid_error[0, 1] = self.compute_rmse()
        valid_error[0, 2] = self.compute_psnr()
        valid_error[0, 3] = self.compute_ergas()
        valid_error[0, 4] = self.sam()
        return valid_error

    def compute_mrae(self):
        error = np.abs(self.output - self.label) / self.label
        # error = torch.abs(outputs - label)
        mrae = np.mean(error.reshape(-1))
        return mrae

    def compute_rmse(self):
        rmse = np.sqrt(np.mean((self.label-self.output)**2))
        return rmse

    def compute_psnr(self):
        
        assert self.label.ndim == 3 and self.output.ndim == 3

        img_c, img_w, img_h = self.label.shape
        ref = self.label.reshape(img_c, -1)
        tar = self.output.reshape(img_c, -1)
        msr = np.mean((ref - tar) ** 2, 1)
        max1 = np.max(ref, 1)

        psnrall = 10 * np.log10(1 / msr)
        out_mean = np.mean(psnrall)
        # return out_mean, max1
        return out_mean

    def compute_ergas(self, scale=8):
        d = self.label - self.output
        ergasroot = 0
        for i in range(d.shape[0]):
            ergasroot = ergasroot + np.mean(d[i, :, :] ** 2) / np.mean(self.label[i, :, :]) ** 2
        ergas = (100 / scale) * np.sqrt(ergasroot/(d.shape[0]+1))
        return ergas

    def compute_sam(self):
        assert self.label.ndim == 3 and self.label.shape == self.label.shape

        c, w, h = self.label.shape
        x_true = self.label.reshape(c, -1)
        x_pred = self.output.reshape(c, -1)

        x_pred[np.where((np.linalg.norm(x_pred, 2, 1)) == 0),] += 0.0001

        sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))

        sam = np.arccos(sam) * 180 / np.pi
        # sam = np.arccos(sam)
        mSAM = sam.mean()
        var_sam = np.var(sam)
        # return mSAM, var_sam
        return mSAM

    def compute_ssim(self, data_range=1, multidimension=False):
        """
        :param x_true:
        :param x_pred:
        :param data_range:
        :param multidimension:
        :return:
        """
        mssim = [
            compare_ssim(X=self.label[i, :, :], Y=self.output[i, :, :], data_range=data_range, multidimension=multidimension)
            for i in range(self.label.shape[0])]

        return np.mean(mssim)


    def ssim(self):
        fout_0 = np.transpose(self.output, [1,2,0])
        hsi_g_0 = np.transpose(self.label, [1,2,0])
        ssim_result = compare_ssim(fout_0, hsi_g_0)
        return ssim_result
    
    def psnr(self):
        fout = self.output*(2**16-1)
        hsi_g = self.label*(2**16-1)
        psnr_g = []
        for i in range(31):
            psnr_g.append(skm.compare_psnr(hsi_g[i,:,:],fout[i,:,:],(2**16-1)))
        return np.mean(np.array(psnr_g))

    def sam(self):
        """
        Compute SAM between two images
        :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
        :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
        :return: Spectral Angle Mapper between `recovered` and `groundTruth`.
        """
        groundTruth = np.transpose(self.label, [1,2,0])
        recovered = np.transpose(self.output, [1,2,0])
        assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

        nom = np.sum(groundTruth * recovered, 2)
        denom1 = np.sqrt(np.sum(groundTruth**2, 2))
        denom2 = np.sqrt(np.sum(recovered ** 2, 2))
        sam = np.arccos(np.divide(nom, denom1*denom2))
        sam = np.divide(sam, np.pi) * 180.0
        sam = np.mean(sam)

        return sam
