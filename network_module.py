import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from CAP_channel import *
from mmcv.ops.deform_conv import DeformConv2d
# ----------------------------------------
#               Conv2d Block
# ----------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False, scale_factor = 2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.conv2d(x)
        return x

class ResConv2dLayer(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False, scale_factor = 2):
        super(ResConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.conv2d = nn.Sequential(
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn),
            Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation = 'none', norm = norm, sn = sn)
        )
    
    def forward(self, x):
        residual = x
        out = self.conv2d(x)
        out = 0.1 * out + residual
        return out

class ResidualDenseBlock_1(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(ResidualDenseBlock_1, self).__init__()
        # dense convolutions
        # self.conv1_1 = Conv2dLayer(latent_channels, latent_channels, 1, stride, 0, dilation, pad_type, activation, norm, sn)
        # self.conv1_2 = Conv2dLayer(latent_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        # self.conv2 = Conv2dLayer(latent_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.conv3 = Conv2dLayer(latent_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        # self.conv4 = Conv2dLayer(latent_channels*2, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        # self.conv5 = Conv2dLayer(latent_channels*2, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)

        # self.conv6 = Conv2dLayer(latent_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        # self.conv7 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        # self.cspn1_guide = Conv2D(latent_channels)
        # self.cspn1 = Affinity_Propagate_Spatial()

    def forward(self, x):
        x1_1 = F.avg_pool2d(x,(8,8),(8,8))
        x1_1 = F.interpolate(x1_1, scale_factor=8)
        x1_1 = x-x1_1
        x1_1 = F.sigmoid(x1_1)
        x1 = x1_1*x
        # x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        # guidance1 = self.cspn1_guide(x3)
        # x3_2 = self.cspn1(guidance1, x3)
        # x4 = self.conv4(torch.cat((x3, x3_2), 1))
        # x5 = self.conv5(torch.cat((x2, x4), 1))
       
        # x5 = self.conv6(x4)
        # x5 = x5+residual
        # x5 = self.conv7(x5)
        return x3

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(ResidualDenseBlock_5C, self).__init__()
        # dense convolutions
        self.conv1 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv2 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv3 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv4 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv5 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv6 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.cspn2_guide = GMLayer(in_channels)
        self.cspn2 = Affinity_Propagate_Channel()
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        guidance2 = self.cspn2_guide(x3)
        x3_2 = self.cspn2(guidance2, x3)
        x4 = self.conv4(torch.cat((x3, x3_2), 1))
        x5 = self.conv5(torch.cat((x2, x4), 1))
        x6 = self.conv6(torch.cat((x1, x5), 1))
        return x6
        


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

class ResidualDenseBlock_5C1(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super(ResidualDenseBlock_5C1, self).__init__()
        self.conv1 = conv3x3(in_channels, latent_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels, latent_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.add(out, residual)
        return out
# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

# ----------------------------------------
#           Spectral Norm Block
# ----------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name = 'weight', power_iterations = 1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

# ----------------------------------------
#             Non-local Block
# ----------------------------------------
class Self_Attn(nn.Module):
    """ Self attention Layer for Feature Map dimension"""
    def __init__(self, in_dim, latent_dim = 8):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Height * Width)
        """
        batchsize, C, height, width = x.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query  = self.query_conv(x).view(batchsize, -1, height * width).permute(0, 2, 1)
        # proj_query: reshape to B x c x N, N = H x W
        proj_key =  self.key_conv(x).view(batchsize, -1, height * width)
        # transpose check, energy: B x N x N, N = H x W
        energy =  torch.bmm(proj_query, proj_key)
        # attention: B x N x N, N = H x W
        attention = self.softmax(energy)
        # proj_value is normal convolution, B x C x N
        proj_value = self.value_conv(x).view(batchsize, -1, height * width)
        # out: B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, height, width)
        
        out = self.gamma * out + x
        return out

# ----------------------------------------
#              Global Block
# ----------------------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GlobalBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False, reduction = 8):
        super(GlobalBlock, self).__init__()
        self.conv1 = Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.conv2 = Conv2dLayer(in_channels, in_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_channels // reduction, in_channels // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_channels // reduction, in_channels, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # residual
        residual = x
        # Sequeeze-and-Excitation(SE)
        b, c, _, _ = x.size()
        x = self.conv1(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = self.conv2(x)
        # addition
        out = 0.1 * y + residual
        return out


class SFT_Layer1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SFT_Layer1, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 1)

        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=self.opt.scale, mode='bilinear')

        x = F.leaky_relu(self.conv1(x), inplace=True)
        return x


class SFT_Layer2(nn.Module):
    def __init__(self, in_channel):
        super(SFT_Layer2, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv0 = nn.Conv2d(in_channel*2, in_channel, 3)
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3)
        self.conv2 = nn.Conv2d(in_channel, in_channel, 3)
        self.conv3 = nn.Conv2d(in_channel, in_channel, 3)
        self.conv4 = nn.Conv2d(in_channel, in_channel, 3)


    def forward(self, x, y):
        x1 = torch.cat((x, y), 1)
        x1 = F.leaky_relu(self.conv0(self.pad(x1)), inplace=True)
        x1 = F.leaky_relu(self.conv1(self.pad(x1)), inplace=True)
        x1 = y*x1+x
        # x2 = F.leaky_relu(self.conv3(self.pad(x)), inplace=True)
        # # x2 = F.leaky_relu(self.conv4(self.pad(x2)), inplace=True)
        # x2 = self.conv4(self.pad(x2))
        # out = y*x1+0.5*x2
        return x1


class DeformableConvBlock(nn.Module):
    def __init__(self, input_channels):
        super(DeformableConvBlock, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels=input_channels * 2, out_channels=1 * 2 * 9, kernel_size=3, padding=1,
                                     bias=True)
        self.deformconv = DeformConv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3,
                                       padding=1)

    def forward(self, lr_features, hr_features):
        input_offset = torch.cat((lr_features, hr_features), dim=1)
        estimated_offset = self.offset_conv(input_offset)
        output = self.deformconv(x=hr_features, offset=estimated_offset)
        return output