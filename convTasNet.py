# -*- encoding: utf-8 -*-
'''
@Filename    :model.py
@Time        :2020/07/09 18:22:54
@Author      :Kai Li
@Version     :1.0
'''

import os
import torch
#from Loss import Loss
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from Datasets import WSJ0Dataset
import pytorch_lightning as pl
EPS = 1e-8

class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = x.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(x-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)

        gLN_y = self.gamma * (x - mean) / torch.pow(var + EPS, 0.5) + self.beta

        return gLN_y

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def select_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D_Block(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, dilation=1, norm_type='gLN'):
        super(Conv1D_Block, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm_type, out_channels)
        if norm_type == 'gLN':
            self.padding = (dilation*(kernel_size-1))//2
        else:
            self.padding = dilation*(kernel_size-1)
        self.dwconv = nn.Conv1d(out_channels, out_channels, kernel_size,
                                1, dilation=dilation, padding=self.padding, groups=out_channels, bias=True)#    padding以保持same
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm_type, out_channels)
        self.sconv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.norm_type = norm_type

    def forward(self, x):
        w = self.conv1x1(x) #   输入【128，L】，输出【512，L】
        w = self.norm1(self.prelu1(w))
        w = self.dwconv(w)
        if self.norm_type == 'cLN':
            w = w[:, :, :-self.padding]
        w = self.norm2(self.prelu2(w))
        w = self.sconv(w)   #   输入【512，L】，输出【128，L】
        x = x + w           #   返回 x + f(x) 
        return x


class TCN(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, norm_type='gLN', X=8):
        super(TCN, self).__init__()
        seq = []
        for i in range(X):
            seq.append(Conv1D_Block(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, norm_type=norm_type, dilation=2**i))
        self.tcn = nn.Sequential(*seq)

    def forward(self, x):
        return self.tcn(x)  #   输入【128，frames】，输出【128，frames】


class Separation(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, norm_type='gLN', X=8, R=3):
        super(Separation, self).__init__()
        s = [TCN(in_channels=in_channels, out_channels=out_channels,
                 kernel_size=kernel_size, norm_type=norm_type, X=X) for i in range(R)]  #输入【128，frames】，输出【128，frames】
        self.sep = nn.Sequential(*s)

    def forward(self, x):
        return self.sep(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, bottleneck=128, kernel_size=16, norm_type='gLN'):
        super(Encoder, self).__init__()
        self.encoder = nn.Conv1d(
            in_channels, out_channels, kernel_size, kernel_size//2, padding=0)  #   输入【1，samples】,输出【512，frames】
        self.norm = select_norm(norm_type, out_channels)    #   归一化
        self.conv1x1 = nn.Conv1d(out_channels, bottleneck, 1)   #   输入【512，fraems】,输出【128，frames】

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt,推理时，每次只计算一个序列
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        x = self.encoder(x)
        w = self.norm(x)
        w = self.conv1x1(w)
        return x, w         #   w给TCN,    x作为混合信号与mask相乘


class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, kernel_size=16):
        super(Decoder, self).__init__()
        self.decoder = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, kernel_size//2, padding=0, bias=True)

    def forward(self, x):
        x = self.decoder(x)
        return torch.squeeze(x)


class ConvTasNet(pl.LightningModule):
    def __init__(self,N=512,L=16,B=128,H=512,P=3,X=8,R=3,norm="gLN",num_spks=2,activate="relu",causal=False):
        super(ConvTasNet, self).__init__()
        # -----------------------model-----------------------
        self.encoder = Encoder(1, N, B, L, norm)

        self.separation = Separation(B, H, P, norm, X, R)
        self.mask = nn.Conv1d(B, H*num_spks, 1, 1)  #   产生mask

        self.decoder = Decoder(H, 1, L)

        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax
        }
        self.non_linear = supported_nonlinear[activate]
        self.num_spks = num_spks

    def forward(self, x):
        x, w = self.encoder(x)      #   输入[1,samples]  输出x【N(512)，frames】，输出w【B(128)，frames】给TCN
        w = self.separation(w)      #   输入【B(128)，frames】,输出【B(128)，frames】
        m = self.mask(w)            #   输入【B(128)，frames】,输出【512 * src, frames】
        m = torch.chunk(m, chunks=self.num_spks, dim=1)     #   分块
        m = self.non_linear(torch.stack(m, dim=0))
        d = [x*m[i] for i in range(self.num_spks)]          #   乘以mask
        s = [self.decoder(d[i]) for i in range(self.num_spks)]  #   decoder，输入【H,frame】，输出【L，frame】
        return s

def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6

if __name__ == "__main__":
    conv = ConvTasNet()
    a = torch.randn(4, 320)
    s = conv(a)
    print(check_parameters(conv))
