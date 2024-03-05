import torch
import torch.nn as nn
import torch.nn.functional as F
from network_module import *


# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                 Generator
# ----------------------------------------
class SGN(nn.Module):
    def __init__(self, opt):
        super(SGN, self).__init__()
        self.bot1 = Conv2dLayer(3, opt.start_channels, 1, 1, 0, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.bot31 = ResidualDenseBlock_1(opt.in_channels, opt.start_channels)
        self.main1 = Conv2dLayer(opt.in_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.main31 = ResidualDenseBlock_5C(opt.start_channels, opt.start_channels)
        self.main32 = ResidualDenseBlock_5C(opt.start_channels, opt.start_channels)
        self.main33 = ResidualDenseBlock_5C(opt.start_channels, opt.start_channels)
        self.main34 = ResidualDenseBlock_5C(opt.start_channels, opt.start_channels)
        self.main4 = Conv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.defconv = DeformableConvBlock(opt.start_channels)


    def forward(self, x, x_hyper):
        x_hyper = torch.nn.functional.interpolate(x_hyper, scale_factor=(8,8), mode='bilinear')
        residule = x_hyper
        x_hyper = self.main1(x_hyper)                                     # out: batch * 32 * 256 * 256

        x = self.bot1(x)                                      # out: batch * 64 * 128 * 128
        spatial_feture = self.bot31(x)                                     # out: batch * 64 * 128 * 128
        x_hyper = x_hyper+x
        x_hyper = self.main31(x_hyper)                                      # out: batch * 32 * 256 * 256
        x_hyper = self.defconv(spatial_feture, x_hyper)
        x_hyper = self.main32(x_hyper)                                      # out: batch * 32 * 256 * 256
        x_hyper = self.defconv(spatial_feture, x_hyper)
        x_hyper = self.main33(x_hyper)                                      # out: batch * 32 * 256 * 256
        x_hyper = self.defconv(spatial_feture, x_hyper)
        x_hyper = self.main34(x_hyper)                                      # out: batch * 32 * 256 * 256
        x_hyper = self.defconv(spatial_feture, x_hyper)
        x_hyper = self.main4(x_hyper)                                       # out: batch * 3 * 256 * 256
        x_hyper = x_hyper+residule
        return x_hyper



