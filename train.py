import torch
from network_code1 import SGN
import argparse
from torch.nn.init import xavier_normal_ as x_init
import itertools
from torch.utils import data
from dataset import HyperDatasetTrain1, HyperDatasetValid
from torchvision.utils import make_grid
import numpy as np
import os
import time
from torch.utils.data import DataLoader
import skimage.measure as skm
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import skimage as ski
import random
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import warnings
from torch.autograd import Variable

warnings.filterwarnings("ignore")
# port = 8100
now = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
os.mkdir('save_model'+now)
# os.mkdir(now)
def xavier_init_(model):
    for name,param in model.named_parameters():
        if 'weight' in name:
            x_init(param)

def loss_spa(hsi,hsi_t,Mse):
    loss = Mse(hsi[1],hsi_t[1]) + Mse(hsi[2],hsi_t[2])
    return loss

def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()

def save_model(epoch,Renet):
    model_state_dict = {'spenet':Renet.state_dict()}
    os.makedirs('./save_model'+now,exist_ok=True)
    torch.save(model_state_dict,'./save_model'+now+'/state_dicr_{}.pkl'.format(epoch))

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

def get_spe_gt(hsi):
    output=[]
    output.append(hsi)
    index = [np.array(list(range(8)))*4,np.array(list(range(16)))*2]
    index[0][-1] = 30
    index[1][-1] = 30
    output.append(hsi[:,index[0],:,:])
    output.append(hsi[:,index[1],:,:])
    return output

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def record_loss(loss_csv,epoch, psnr, ssim):
    """ Record many results."""
    loss_csv.write('{},{},{}\n'.format(epoch, psnr, ssim))
    loss_csv.flush()    
    loss_csv.close
conti = False
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parse = argparse.ArgumentParser()
    parse.add_argument('--n_epochs', type = int, default = 1400, help = 'number of epochs of training')
    parse.add_argument('--batch_size',type = int, default = 4, help = 'size of the batches')
    parse.add_argument('--lr',type = float, default = 1e-3*0.25, help='learing rate of network')
    parse.add_argument('--b1',type = float, default = 0.9, help='adam:decay of first order momentum of gradient')
    parse.add_argument('--b2',type = float, default = 0.999, help= 'adam: decay of first order momentum of gradient')
    parse.add_argument('--envname',type = str, default = 'temp', help= 'adam: decay of first order momentum of gradient')
    parse.add_argument('--local_rank', type=int, default=0)
    parse.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parse.add_argument('--activ', type = str, default = 'lrelu', help = 'activation type of networks')
    parse.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parse.add_argument('--in_channels', type = int, default = 31, help = 'input channels for generator')
    parse.add_argument('--out_channels', type = int, default = 31, help = 'output channels for generator')
    parse.add_argument('--start_channels', type = int, default = 48, help = 'start channels for generator')
    parse.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parse.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')

    setup_seed(50)
    factor_lr = 1
    opt = parse.parse_args()
    loss_csv = open(os.path.join('./', 'save_model'+now, 'loss.csv'), 'a+')
    Hyper_test = HyperDatasetValid()
    Hyper_test = DataLoader(Hyper_test,batch_size=1,shuffle=False, num_workers=4,pin_memory=False,
                                     drop_last=True)

    Hyper_train = HyperDatasetTrain1()
    datalen = Hyper_train.__len__()
    Hyper_train = DataLoader(Hyper_train,batch_size=opt.batch_size,shuffle=False, num_workers=10,pin_memory=False,
                                     drop_last=True)
    spenet = SGN(opt).cuda()
    optimzier = torch.optim.Adam(itertools.chain(spenet.parameters()),lr = opt.lr,betas=(opt.b1,opt.b2),weight_decay=0)
    # MseLoss = torch.nn.MSELoss()
    MaeLoss = torch.nn.L1Loss().cuda()  
    T_max = (datalen//(1*opt.batch_size))*1000
    print(T_max)
    schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzier, T_max, eta_min=1e-5*0.1, last_epoch=-1) 
    b_r = 0
    ssim_best = 0
    psnr_best = 0

    for epoch in (range(opt.n_epochs)):
        batch = 0
        loss_ = []
        loss_t = []
        ssim_log = []
        for msi, hsi_g, hsi in tqdm(Hyper_train):
            msi = msi.cuda()
            hsi_g = hsi_g.cuda()
            hsi = hsi.cuda()
            msi = Variable(msi)
            hsi_g = Variable(hsi_g)
            hsi = Variable(hsi)
            batch = batch+1
            b_r = max(batch,b_r)
            spenet.train()
            refined = spenet(msi, hsi)
            optimzier.zero_grad()
            loss = MaeLoss(refined,hsi_g)
            loss.backward()
            optimzier.step()
            schduler.step()
            loss_.append(loss.item())
            del loss
        a = np.random.randint(0,31,refined.shape[0])
        a_ = np.array(range(refined.shape[0]))
        output = refined.detach().cpu()[a_,a,:,:]*255
        output = output[:,None,:,:]
        output = torch.cat([output,output,output],1)
        output_ = make_grid(output,nrow=int(5)).numpy()
        psnr_g = []
        if (epoch>800) or (epoch % 20 == 0):
            with torch.no_grad():
                for (msi, hsi_g, hsi, _) in tqdm(Hyper_test):
                    spenet.eval()
                    msi = msi.cuda()
                    hsi_g = hsi_g.cuda()
                    hsi = hsi.cuda()
                    msi = Variable(msi)
                    hsi_g = Variable(hsi_g)
                    hsi = Variable(hsi)
                    refined = spenet(msi, hsi)
                    fout = refined
                    hsi_g_ = (hsi_g).detach().cpu().numpy()
                    fout_ = (fout).detach().cpu().numpy()
                    for i in range(31):
                        psnr_g.append(compare_psnr(hsi_g_[0,i,:,:],fout_[0,i,:,:]))
                    fout_0 = np.transpose(fout_,(0,2,3,1))[0,:,:,:]
                    hsi_g_0 = np.transpose(hsi_g_,(0,2,3,1))[0,:,:,:]
                    ssim = compare_ssim(fout_0, hsi_g_0, K1 = 0.01, K2 = 0.03, multichannel=True)
                    ssim_log.append(ssim)
            ssim_ = np.mean(np.array(ssim_log))
            psnr_ = np.mean(np.array(psnr_g))
            record_loss(loss_csv, epoch, psnr_, ssim_)
            print("epoch:%d, train_loss:%f, psnr:%f, ssim:%f"%(epoch, loss_[-1], psnr_, ssim_))
            if ssim_ > 0.999 and epoch > 0:
                save_model(epoch,spenet)
            if psnr_ > 51.50 and epoch > 0:
                save_model(epoch,spenet)

            if epoch % 100 == 0:
                save_model(epoch,spenet)
    save_model(epoch,spenet)