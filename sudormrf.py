"""!
@brief SuDO-RM-RF model

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import torch
import torch.nn as nn
import math

from torch.nn.modules.normalization import _shape_t

class ChannelLayerNorm(nn.LayerNorm):                                          # layernorm, 对最后几个维度进行归一化;输入为维度的大小，此处为enc_num_basis
    def __init__(self, dim, eps = 1e-5,elementwise_affine=True):
        super(ChannelLayerNorm, self).__init__(
            dim, eps,elementwise_affine=elementwise_affine)
    def forward(self,x):
        x = x.permute(0,2,1)
        y= super().forward(x)
        y = y.permute(0,2,1)
        return y
    
class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=256,
                      kernel_size=256,
                      stride=21 // 2,
                      padding=21 // 2) -> None:
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential(*[nn.Conv1d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride= stride,
                                                 padding= padding),
                                                 nn.ReLU()])
    
    def forward(self,x):
        return self.encoder(x)


class ConvNormAct(nn.Module):
    '''
    This class defines the convolution layer with normalization and a PReLU
    activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    '''
    This class defines the convolution layer with normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    '''
    This class defines a normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: number of output channels
        '''
        super().__init__()
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    '''
    This class defines the dilated convolution with normalized output.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class UBlock(nn.Module):
    '''
    This class defines the Upsampling block, which is based on the following
    principle:
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self,
                 out_channels=128,
                 in_channels=256,
                 upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1,               #   逐点卷积变换维度到in_channels
                                    stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5,   #   膨胀卷积层 1，d = 0,输入输出尺寸不变
                                           stride=1, groups=in_channels, d=1))

        for i in range(1, upsampling_depth):                                    #   逐层卷积，通过stride = 2 实现下采样
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels,
                                               kSize=2*stride + 1,
                                               stride=stride,
                                               groups=in_channels, d=1))
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(scale_factor=2,
                                               # align_corners=True,
                                               # mode='bicubic'
                                               )
        self.conv_1x1_exp = ConvNorm(in_channels, out_channels, 1, 1, groups=1)
        self.final_norm = NormAct(in_channels)
        self.module_act = NormAct(out_channels)

    def forward(self, x):

        output1 = self.proj_1x1(x)                          #   逐点卷积变换特征大小，[batch, out_chan,K] -> [batch, in_chan,K] 

        output = [self.spp_dw[0](output1)]                  #   第0层，不做降采样，[batch, in_chan,K] -> [batch, in_chan,K], list
        for k in range(1, self.depth):                      #   逐层卷积，通过stride = 2 降采样
            out_k = self.spp_dw[k](output[-1])                  #   [batch, in_chan,K]
            output.append(out_k)                                #   k -> k/2 -> k/4 -> k/8

        for _ in range(self.depth-1):                       #   上采样
            resampled_out_k = self.upsampler(output.pop(-1))    #   k/8 -> k/4 -> k/2 -> k   --->    k/4 -> k/2 -> k
            output[-1] = output[-1] + resampled_out_k           #   特征融合是通过相加实现，计算量小，效率高；在不需要精确恢复原图像时可以用

        #   K                   +               K
        #       K/2             +           K/2
        #           K/4         +       K/4    
        #               K/8     ->

        #   norm & point-wise 卷积
        expanded = self.conv_1x1_exp(self.final_norm(output[-1]))   #层归一化，激活；变换维度，激活并输出
        #   返回残差连接，注意expanded和数据x的尺度要一致
        return self.module_act(expanded + x)                #   残差连接，层归一化，激活


class SuDORMRF(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=256,
                 num_blocks=16,
                 upsampling_depth=4,
                 enc_kernel_size=21,
                 enc_num_basis=256,
                 num_sources=2):
        super(SuDORMRF, self).__init__()

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2 **
                       self.upsampling_depth) // math.gcd(
                       self.enc_kernel_size // 2,
                       2 ** self.upsampling_depth)

        # encoder,stride为kernel的一半，padding为kernel的一半，保证输出长度不变
        self.encoder = Encoder(in_channels=1, out_channels=enc_num_basis,
                      kernel_size=enc_kernel_size,
                      stride=enc_kernel_size // 2,
                      padding=enc_kernel_size // 2)
        
        # Separation module
        self.layerNorm = ChannelLayerNorm(enc_num_basis, eps=1e-08)

        self.l1 = nn.Conv1d(in_channels=enc_num_basis,
                            out_channels=out_channels,
                            kernel_size=1)
        self.separator = nn.Sequential(*[
            UBlock(out_channels=out_channels,
                   in_channels=in_channels,
                   upsampling_depth=upsampling_depth)
            for r in range(num_blocks)])

        if out_channels != enc_num_basis:
            self.reshape_before_masks = nn.Conv1d(in_channels=out_channels,
                                                  out_channels=enc_num_basis,
                                                  kernel_size=1)
        # Masks layer
        self.mask = nn.Conv2d(in_channels=1,
                           out_channels=num_sources,
                           kernel_size=(enc_num_basis + 1, 1),
                           padding=(enc_num_basis - enc_num_basis // 2, 0))

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources,
            out_channels=num_sources,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=num_sources)
        #self.ln_mask_in = nn.GroupNorm(1, enc_num_basis, eps=1e-08)
    # Forward pass
    def forward(self, input_wav):

        #   [batch,samples] -> [batch,1,samples]
        #print("input_wav shape {}.f4")
        input_wav = input_wav.unsqueeze(1)
        # 1  前端处理
        x = self.pad_to_appropriate_length(input_wav)   #   开头padding，到self.lcm的整数倍,保证下采样过程中不会出现奇数
        x = self.encoder(x)                             #   [batch,1,samples] -> [batch,enc_num, K]
        s = x.clone()                                   #   备份

        # 2  分离模块
        x = self.layerNorm(x)                           #   对特征进行归一化， [batch,enc_num, K] -> [batch,enc_num, K]
        x = self.l1(x)                                  #   逐点卷积变换尺度到out_channels， [batch,enc_num, K] -> [batch,out_channels, K]
        x = self.separator(x)                           #   U-net，0.25x，连续做4次
        if self.out_channels != self.enc_num_basis:     
            x = self.reshape_before_masks(x)

        # 3  计算mask，施加
        x = self.mask(x.unsqueeze(1))
        if self.num_sources == 1:
            x = torch.sigmoid(x)
        else:
            x = nn.functional.softmax(x, dim=1)
        x = x * s.unsqueeze(1)

        # 4 decoder，输出
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        return self.remove_trailing_zeros(estimated_waveforms, input_wav)

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1]) +
                [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=torch.float32,device='cuda')
            padded_x[..., :x.shape[-1]] = x
            return padded_x
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]


if __name__ == "__main__":
    model = SuDORMRF(out_channels=128,
                     in_channels=512,
                     num_blocks=16,
                     upsampling_depth=4,
                     enc_kernel_size=21,
                     enc_num_basis=512,
                     num_sources=2)

    dummy_input = torch.rand(3, 1, 32079)
    estimated_sources = model(dummy_input)
    print(estimated_sources.shape)




