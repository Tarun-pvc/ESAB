"""
Please note that code cleaning and standardization has not been done yet. A lot of this code is hence inaccurately named, and is not functional. 
Some comments have been put in for understanding. 
The full, working code will be put up shortly; Until then, this code might give a good indication of what has been implemented.  
Please reach out and raise an issue. Conversation and questions are appreciated.   
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import numpy as np
from data.GMOD_Dataset import GMOD_Dataset
import argparse
from eval import PSNR, SSIM, SAM
# from skimage.metrics import structural_similarity as SSIM
import math 
from logging_utils import *

import logging


# Model Definition
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        
class NonLocalConvBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.g = nn.Conv3d(in_channels = in_channels, out_channels= in_channels//2 , kernel_size= 1)
        self.phi = nn.Conv3d(in_channels = in_channels, out_channels= in_channels//2, kernel_size=1)
        self.theta = nn.Conv3d(in_channels, out_channels = in_channels//2 , kernel_size = 1)
        self.conv3d = nn.Conv3d(in_channels = in_channels // 2, out_channels= in_channels, kernel_size= 1)
        # self.conv3d_1 = nn.Conv3d(in_channels=in_channels //2, out_channels=in_channels//2, kernel_size = (1, 3, 3), padding =(0, 1, 1))
        # self.conv3d_2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels, kernel_size = (3, 1, 1), padding= (1, 0 , 0))

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        B, C, H, W = x.shape
        
        inter_channels = C // 2

        
        theta = self.theta(x.unsqueeze(-1))
        phi = self.phi(x.unsqueeze(-1))

        theta = theta.reshape(-1, inter_channels)
        phi = phi.reshape(inter_channels, -1)

        theta_phi_matmul = torch.nn.functional.softmax(torch.matmul(theta, phi), dim = 1)

        g = self.g(x.unsqueeze(-1))
        g = g.reshape(-1, inter_channels)

        theta_phi_g_matmul = torch.nn.functional.softmax(torch.matmul(theta_phi_matmul, g), dim = 1)
        theta_phi_g_matmul = theta_phi_g_matmul.reshape(B, inter_channels, H, W)

        final = self.conv3d((theta_phi_g_matmul.unsqueeze(-1))).squeeze(-1) + x

        return final

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()

        self.channels = channels
        self.size = size
        
        # Ensure the number of channels is divisible by the number of heads
        self.num_heads = 4
        self.head_dim = channels // self.num_heads
        self.embed_dim = self.head_dim * self.num_heads
        
        self.mha = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        self.ln = nn.LayerNorm([self.embed_dim])

        self.ff_self = nn.Sequential(
            nn.LayerNorm([self.embed_dim]),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # If channels is not divisible by num_heads, we need to project it
        self.project_in = nn.Linear(channels, self.embed_dim) if channels != self.embed_dim else nn.Identity()
        self.project_out = nn.Linear(self.embed_dim, channels) if channels != self.embed_dim else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2)  # (b, h*w, c)
        
        x = self.project_in(x)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        attention_value = self.project_out(attention_value)
        
        return attention_value.transpose(1, 2).view(b, c, h, w)
  

class Piece3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # 1x3x3 convolution
        self.conv_1x3x3 = nn.Conv3d(in_channels, out_channels, 
                                    kernel_size=(1, kernel_size, kernel_size),
                                    stride=stride, 
                                    padding=(0, padding, padding))
        
        # 3x1x1 convolution
        self.conv_3x1x1 = nn.Conv3d(in_channels, out_channels, 
                                    kernel_size=(kernel_size, 1, 1),
                                    stride=stride, 
                                    padding=(padding, 0, 0))

    def forward(self, x):
        # Perform 1x3x3 convolution
        x = x.unsqueeze(-1)
        out_1x3x3 = self.conv_1x3x3(x)
        
        # Perform 3x1x1 convolution
        out_3x1x1 = self.conv_3x1x1(x)
        
        # Elementwise addition of the two convolution results
        out = out_1x3x3 + out_3x1x1
        out = out.squeeze(-1)
        return out
   

class spefe(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.piece1 = Piece3DConv(in_channels, in_channels)
        # self.leaky_relu = F.leaky_relu()
        self.piece2 = Piece3DConv(in_channels, in_channels)
    
    def forward(self, x):
        residual = x
        h = x.clone()
        h = self.piece1(h)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.piece2(h)
        h = F.leaky_relu(h, negative_slope=0.01)
        h += residual

        return h


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              padding=dilation_rate, dilation=dilation_rate)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
"""

class ESAB(nn.Module):
    def __init__(self, in_channels, args, use_nonlocal=False, use_spefe = False):
        super().__init__()
        self.use_nonlocal = use_nonlocal

        self.conv2d1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.conv2d2 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)


        self.dilatedconv2d1 = DilatedConvBlock(in_channels, in_channels, kernel_size=3, dilation_rate=1)
        self.dilatedconv2d2 = DilatedConvBlock(in_channels, in_channels, kernel_size=3, dilation_rate=2)
        
        self.use_spefe = args.use_spefe
        self.use_nonlocal = args.use_nonlocal
        self.use_rcab = args.use_rcab
        self.use_dilated_conv = args.use_dilated_conv
        self.use_custom_multiscale = args.use_custom_multiscale
        # self.use_eca = args.use_eca

        self.conv3d = Piece3DConv(in_channels, in_channels)

        self.eca_layer1 = eca_layer(in_channels)
        self.eca_layer2 = eca_layer(in_channels)
        self.eca_layer3 = eca_layer(in_channels)

        self.conv3d1 = nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0))
        self.conv3d2 = nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0))
        self.conv3d3 = nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0))
        self.conv3d4 = nn.Conv3d(in_channels, in_channels, kernel_size=(5,1,1), padding=(2,0,0))
        self.conv3d5 = nn.Conv3d(in_channels, in_channels, kernel_size=(5,1,1), padding=(2,0,0))
        self.conv3d6 = nn.Conv3d(in_channels, in_channels, kernel_size=(5,1,1), padding=(2,0,0))

        self.conv3d7 = nn.Conv3d(in_channels, in_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0))

        self.spefe = spefe(in_channels)

        self.nonlocalconv = NonLocalConvBlock(in_channels)

    def forward(self, F_in):
        h = F_in.clone()

        if self.use_dilated_conv:
            h = self.dilatedconv2d1(h)
            h = self.dilatedconv2d2(h)
        else:
            h = self.conv2d1(h)
            h = self.conv2d2(h)
        # h = F.relu(h)
        # h = F.relu(h)
        

        if self.use_custom_multiscale:

            conv3image = F.leaky_relu(self.conv3d1(h.unsqueeze(-1)), negative_slope=0.01)
            conv3image = self.eca_layer1(conv3image.squeeze(-1))

            conv5image = F.leaky_relu(self.conv3d4(conv3image.unsqueeze(-1)), negative_slope=0.01)
            conv5image = self.eca_layer2(conv5image.squeeze(-1))
            
            conv5image = F.leaky_relu(self.conv3d7(conv5image.unsqueeze(-1)), negative_slope=0.01)

            if self.use_spefe:
                h = self.spefe(h)
            elif self.use_rcab:
                rcab = self.conv3d(h)
                rcab = F.leaky_relu(rcab, negative_slope=0.01)
                rcab = self.conv3d(rcab)
                h = self.eca_layer3(rcab) + h

            F_mid = F.softmax(conv5image.squeeze(-1), dim = 1)
            F_out = F_mid * h + F_in
            return F_out
            
            



        else:
            conv3image = F.leaky_relu(self.conv3d1(h.unsqueeze(-1)), negative_slope=0.01)
            conv3image = F.leaky_relu(self.conv3d2(conv3image), negative_slope=0.01)
            conv3image = F.leaky_relu(self.conv3d3(conv3image), negative_slope=0.01)

            conv5image = F.leaky_relu(self.conv3d4(h.unsqueeze(-1)), negative_slope=0.01)
            conv5image = F.leaky_relu(self.conv3d5(conv5image), negative_slope=0.01)
            conv5image = F.leaky_relu(self.conv3d6(conv5image), negative_slope=0.01)

            if self.use_spefe:
                h = self.spefe(h)
            if self.use_rcab:
                rcab = self.conv3d(h)
                rcab = F.leaky_relu(rcab, negative_slope=0.01)
                rcab = self.conv3d(rcab)
                h = self.eca_layer3(rcab) + h
                
            F_mid = conv3image.squeeze(-1) + conv5image.squeeze(-1)

            F_mid = F.softmax(F_mid, dim=1)
            F_out = F_mid * h + F_in
            return F_out

        # if self.use_nonlocal:
        #     F_mid = self.nonlocalconv(F_mid)

"""



class ESAB(nn.Module):
    def __init__(self, in_channels, args, use_nonlocal=False, use_spefe=False):
        super().__init__()
        
        if args.use_dilated_conv:
            self.dilatedconv2d1 = DilatedConvBlock(in_channels, in_channels, kernel_size=3, dilation_rate=1)
            self.dilatedconv2d2 = DilatedConvBlock(in_channels, in_channels, kernel_size=3, dilation_rate=2)
        else:
            self.conv2d1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            self.conv2d2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        if args.use_custom_multiscale:
            self.conv3d1 = nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0))
            self.conv3d4 = nn.Conv3d(in_channels, in_channels, kernel_size=(5,1,1), padding=(2,0,0))
            self.conv3d7 = nn.Conv3d(in_channels, in_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0))
            self.eca_layer1 = eca_layer(in_channels)
            self.eca_layer2 = eca_layer(in_channels)
        else: 
            self.conv3d1 = nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0))
            self.conv3d2 = nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0))
            self.conv3d3 = nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0))

            self.conv3d4 = nn.Conv3d(in_channels, in_channels, kernel_size=(5,1,1), padding=(2,0,0))
            self.conv3d5 = nn.Conv3d(in_channels, in_channels, kernel_size=(5,1,1), padding=(2,0,0))
            self.conv3d6 = nn.Conv3d(in_channels, in_channels, kernel_size=(5,1,1), padding=(2,0,0))        
        
        if args.use_spefe:
            self.spefe = spefe(in_channels)
        
        if args.use_rcab:
            self.conv3d = Piece3DConv(in_channels, in_channels)
            self.eca_layer3 = eca_layer(in_channels)
        
        if args.use_nonlocal:
            self.nonlocalconv = NonLocalConvBlock(in_channels)
        
        # We register which parts to use in the forward pass. Defined by args passed to the model. 
        # The args are being taken in through command line arguments, using argparse. 
        self.use_dilated_conv = args.use_dilated_conv
        self.use_custom_multiscale = args.use_custom_multiscale
        self.use_spefe = args.use_spefe
        self.use_rcab = args.use_rcab
        self.use_nonlocal = args.use_nonlocal

    def forward(self, F_in):
        h = F_in.clone()

        if self.use_dilated_conv:
            h = self.dilatedconv2d1(h)
            h = self.dilatedconv2d2(h)
        else:
            h = self.conv2d1(h)
            h = self.conv2d2(h)
        # h = F.relu(h)
        # h = F.relu(h)
        

        if self.use_custom_multiscale:

            # conv3image = F.leaky_relu(self.conv3d1(h.unsqueeze(-1)), negative_slope=0.01)
            # conv3image = self.eca_layer1(conv3image.squeeze(-1))

            # conv5image = F.leaky_relu(self.conv3d4(conv3image.unsqueeze(-1)), negative_slope=0.01)
            # conv5image = self.eca_layer2(conv5image.squeeze(-1))
            
            # conv5image = F.leaky_relu(self.conv3d7(conv5image.unsqueeze(-1)), negative_slope=0.01)

            conv7image = F.leaky_relu(self.conv3d7(h.unsqueeze(-1)), negative_slope=0.01)
            conv7image = self.eca_layer1(conv7image.squeeze(-1))

            conv5image = F.leaky_relu(self.conv3d4(conv7image.unsqueeze(-1)), negative_slope=0.01)
            conv5image = self.eca_layer2(conv5image.squeeze(-1))
            
            conv3image = F.leaky_relu(self.conv3d1(conv5image.unsqueeze(-1)), negative_slope=0.01)



            if self.use_spefe:
                h = self.spefe(h)
            if self.use_rcab:
                rcab = self.conv3d(h)
                rcab = F.leaky_relu(rcab, negative_slope=0.01)
                rcab = self.conv3d(rcab)
                h = self.eca_layer3(rcab) + h

            F_mid = F.softmax(conv3image.squeeze(-1), dim = 1)
            F_out = F_mid * h + F_in
            return F_out

        else:
            conv3image = F.leaky_relu(self.conv3d1(h.unsqueeze(-1)), negative_slope=0.01)
            conv3image = F.leaky_relu(self.conv3d2(conv3image), negative_slope=0.01)
            conv3image = F.leaky_relu(self.conv3d3(conv3image), negative_slope=0.01)

            conv5image = F.leaky_relu(self.conv3d4(h.unsqueeze(-1)), negative_slope=0.01)
            conv5image = F.leaky_relu(self.conv3d5(conv5image), negative_slope=0.01)
            conv5image = F.leaky_relu(self.conv3d6(conv5image), negative_slope=0.01)

            if self.use_spefe:
                h = self.spefe(h)
            if self.use_rcab:
                rcab = self.conv3d(h)
                rcab = F.leaky_relu(rcab, negative_slope=0.01)
                rcab = self.conv3d(rcab)
                h = self.eca_layer3(rcab) + h
                
            F_mid = conv3image.squeeze(-1) + conv5image.squeeze(-1)

            F_mid = F.softmax(F_mid, dim=1)
            F_out = F_mid * h + F_in
            return F_out

        # if self.use_nonlocal:
        #     F_mid = self.nonlocalconv(F_mid)

        


class SubNetwork(nn.Module):
    def __init__(self, in_channels, args, upscale_factor=1, use_nonlocal = False):
        super().__init__()
        self.ESAB1 = ESAB(in_channels, args)
        # self.attention1 = SelfAttention(in_channels, l_resolution)
        self.attention1 = eca_layer(in_channels)
        self.ESAB2 = ESAB(in_channels, args)
        # self.attention1 = eca_layer(in_channels
                                    # )
        self.ESAB3 = ESAB(in_channels, args, use_nonlocal)
        self.ESAB4 = ESAB(in_channels, args, use_nonlocal)
        self.ESAB5 = ESAB(in_channels, args,use_nonlocal)
        self.ESAB6 = ESAB(in_channels, args,use_nonlocal)
        self.ESAB7 = ESAB(in_channels, args,use_nonlocal)
        self.ESAB8 = ESAB(in_channels, args,use_nonlocal)
        self.conv = nn.Conv2d(in_channels, in_channels * upscale_factor**2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.relu = nn.ReLU()

    def forward(self, x):
        h = x
        m1 = self.ESAB1(h)
        # m1 = F.relu(m1)
        m2 = self.ESAB2(m1)
        m2 += m1
        m2 = self.relu(m2)
        # m2 = self.attention1(m2)
        m3 = self.ESAB3(m2)
        m3 += m2
        m3 = self.relu(m3)
        # m3 = self.attention1(m3)
        m4 = self.ESAB4(m3)
        m4 += m3
        m4 = self.relu(m4)

        m5 = self.ESAB5(m4)
        m5 += m4
        m5 = self.relu(m5)

        m6 = self.ESAB6(m5)
        m6 += m5
        m6 = self.relu(m6)

        m7 = self.ESAB7(m6)
        m7 = self.relu(m7)
        m8 = self.ESAB8(m7)

        m8 += x
        m8 = self.relu(m8)
        m8 = self.conv(m8)
        m8 = self.pixel_shuffle(m8)
        # m8 = F.relu(m8)
        return m8

class Trunk(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor = 2, use_nonlocal = False):
        super().__init__()
        self.inconv = Piece3DConv(in_channels, in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

        # self.sa1 = SelfAttention(in_channels, l_resolution)
        self.sa1 = eca_layer(in_channels)
        self.sa2 = SelfAttention(in_channels, l_resolution)
        self.sa3 = SelfAttention(in_channels, l_resolution)
        # self.sa4 = SelfAttention(in_channels, l_resolution)
        self.use_nonlocal = use_nonlocal

        # if use_nonlocal:
        #     self.nonlocalconv = NonLocalConvBlock(out_channels)
        
        self.ESAB1 = ESAB(in_channels, args)
        self.ESAB2 = ESAB(in_channels, args)
        self.ESAB3 = ESAB(in_channels, args)
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor**2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    # Changes made: 
    def forward(self, x):
        h = self.inconv(x)
        h = self.conv1(h)
        # h = F.relu(h)
        m1 = self.ESAB1(h)
        # m1 = F.relu(m1)
        # m1 = self.sa1(m1)
        m2 = self.ESAB2(m1)
        # m2 = F.relu(m2)
        m2 = self.sa2(m2)
        m3 = self.ESAB3(m2)
        # m3 = F.relu(m3)
        m3 = self.sa3(m3)
        
        m3 += x
        m3 += m2
        m3 += m1
        m3 = self.conv2(m3)
        
        # if self.use_nonlocal:
        #     m3 = self.nonlocalconv(m3)

        h = m3
        h = self.conv(h)
        h = self.pixel_shuffle(h)
        # h = F.relu(h)
        # h = self.sa4(h)
        return h

class SFEB(nn.Module):
    def __init__(self, num_bands, upscale_factor, args, use_nonlocal = False):
        super().__init__()
        self.use_nonlocal = use_nonlocal
        self.number_of_groups = 6
        self.bands_per_group = num_bands // self.number_of_groups
        self.group_nets = nn.ModuleList([
            SubNetwork(self.bands_per_group, args, use_nonlocal=use_nonlocal) for _ in range(self.number_of_groups)
        ])
        self.trunk = Trunk(self.number_of_groups * self.bands_per_group, num_bands, use_nonlocal = use_nonlocal)
        self.nonlocalblock = NonLocalConvBlock(num_bands)

    def forward(self, x):

        group_outputs = [self.group_nets[i](x[:, i*self.bands_per_group:(i+1)*self.bands_per_group, :, :])
                         for i in range(self.number_of_groups)]
        concatenated_output = torch.cat(group_outputs, dim=1)

        final_output = self.trunk(concatenated_output)
        return final_output

# The base code is the same as my other repository, github.com/Tarun-pvc/GMOD . Hence, the same name. 
class GMOD(nn.Module):
    def __init__(self, num_bands, upscale_factor, args, use_nonlocal = False, use_ECA = False):
        super().__init__()
        self.use_ECA = use_ECA
        self.use_nonlocal = use_nonlocal
        self.sfeb = SFEB(num_bands, upscale_factor, args, use_nonlocal=use_nonlocal)
        self.conv_sr3 = nn.Conv2d(in_channels=3, out_channels=num_bands, kernel_size=3, padding=1, stride=1)
        # self.conv_final1 = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1, stride=1)
        self.conv_final1 = DilatedConvBlock(num_bands, num_bands, kernel_size=3, dilation_rate=4)
        # self.conv_final2 = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1, stride=1)
        # self.conv_final2 = DilatedConvBlock(num_bands, num_bands, kernel_size=3, dilation_rate=2)
        self.conv_final3 = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1, stride=1)
        # self.sr3_weights_path = sr3_weights_path

        if use_ECA: 
            self.sa1 = eca_layer(num_bands)
            # self.sa2 = SelfAttention(num_bands, l_resolution)
            self.sa2 = eca_layer(num_bands)
            self.sa3 = eca_layer(num_bands)
        
        if use_nonlocal:
            self.nonlocalblock = NonLocalConvBlock(num_bands)

        self.upconv = nn.Conv2d(3, num_bands, kernel_size=1, stride = 1)
        self.bicubic_conv = nn.Conv2d(num_bands, num_bands, kernel_size = 3, padding = 1, stride = 1)
    

    def forward(self, x, sr3):
        sfeb_out = self.sfeb(x)
        if self.use_ECA:
            sfeb_out = self.sa1(sfeb_out)

        # sr3_out = SR3(perform_pca(x), opt, weights_path=self.sr3_weights_path)
        # sr3_out = generator(torch.tensor(sfeb_out.cpu(), dtype = (torch.float)))
        # sr3_out = sr3_out.cuda()
        # sr3_out = sr3_out[1].to(device)
        # sr3_out = self.sa2(sr3_out)
        # combined = sfeb_out + 0.01*self.conv_sr3(sr3_out.to("cuda"))
        conv_sr3 = self.upconv(sr3)
        # conv_sr3 = F.relu(conv_sr3)
        combined = sfeb_out + conv_sr3
        # combined = F.relu(combined)

        if self.use_ECA:
            combined = self.sa2(combined)


        smoothed = F.leaky_relu(self.conv_final1(combined), negative_slope=0.01)
        # smoothed = F.leaky_relu(self.conv_final2(smoothed), negative_slope=0.01)
        smoothed = F.leaky_relu(self.conv_final3(smoothed), negative_slope=0.01)

        bicubic_upsampled = F.interpolate(x, scale_factor=scale, mode='bicubic', align_corners=True)

        bicubic_upsampled = self.bicubic_conv(bicubic_upsampled)+bicubic_upsampled

        if self.use_nonlocal:
            bicubic_upsampled = self.nonlocalblock(bicubic_upsampled)
        if self.use_ECA:
            bicubic_upsampled = self.sa3(bicubic_upsampled)

        output = smoothed + bicubic_upsampled
        return output

## CHANGES : NONLOCAL BLOCK NOW USED GLOBALLY, DILATION FACTORS OF THE CONV BLOCKS IN ESAB ARE 1 AND 2
### ========================================================


class customLoss(nn.Module):
    def __init__(self, N, lamd = 1e-1, mse_lamd=1, epoch=None):
        super(customLoss, self).__init__()
        self.N = N
        self.lamd = lamd
        self.mse_lamd = mse_lamd
        self.epoch = epoch
        return

    def forward(self, res, label):
        mse = F.mse_loss(res, label, reduction='sum')
        # mse = func.l1_loss(res, label, size_average=False)
        loss = mse / (self.N * 2)
        esp = 1e-12
        H = label.size()[2]
        W = label.size()[3]
        Itrue = label.clone()
        Ifake = res.clone()
        nom = torch.mul(Itrue, Ifake).sum(dim=1)
        denominator = Itrue.norm(p=2, dim=1, keepdim=True).clamp(min=esp) * \
                      Ifake.norm(p=2, dim=1, keepdim=True).clamp(min=esp)
        denominator = denominator.squeeze()
        # sam = -np.pi/2*torch.div(nom, denominator) + np.pi/2
        sam = torch.div(nom, denominator).acos()
        sam[sam!=sam] = 0
        sam_sum = torch.sum(sam) / (self.N * H * W)
        if self.epoch is None:
            total_loss = self.mse_lamd * loss + self.lamd * sam_sum
        else:
            norm = self.mse_lamd + self.lamd * 0.1 **(self.epoch//10)
            lamd_sam = self.lamd * 0.1 ** (self.epoch // 10)
            total_loss = self.mse_lamd/norm * loss + lamd_sam/norm * sam_sum
        return total_loss


class gradientLoss(nn.Module):
    def __init__(self, N, mse_lambda = 2, gradient_lambda = 2e-1):
        super().__init__()
        self.mse_lambda = mse_lambda
        self.gradient_lambda = gradient_lambda
        self.N = N


    def forward(self, pred, gt, epoch=0):

        mse = F.mse_loss(pred, gt) / (self.N*2)

        pred_diff = torch.diff(pred, dim = 1)
        gt_diff = torch.diff(gt, dim = 1)

        pred_diff_flat = pred_diff.view(-1, pred_diff.shape[1])
        gt_diff_flat = gt_diff.view(-1, gt_diff.shape[1])

        cosine_sim = torch.cosine_similarity(pred_diff_flat, gt_diff_flat, dim = 1)
        slope_loss = 1-cosine_sim
        slope_loss = slope_loss.mean()/self.N

        if epoch == 0: 
            return self.mse_lambda*mse + self.gradient_lambda*slope_loss
        else:
            norm = self.mse_lambda + self.gradient_lambda * 0.1 **(epoch//10)
            lamd_slope = self.gradient_lambda * 0.1 ** (epoch // 10)
            total_loss = self.mse_lambda/norm * mse + lamd_slope/norm * slope_loss
            return self.mse_lambda*mse + self.gradient_lambda*slope_loss


class SpectralDifferenceLoss(nn.Module):
    def __init__(self, N, mse_lambda = 2, gradient_lambda = 2e-1):
        super().__init__()
        self.mse_lambda = mse_lambda
        self.gradient_lambda = gradient_lambda
        self.N = N

    def forward(self, y_true, y_pred, epoch=0):
        # y_true and y_pred should be of shape (batch_size, height, width, num_bands)
        assert y_true.shape == y_pred.shape, "Ground truth and prediction shapes do not match"

        mse = F.mse_loss(y_pred, y_true) / (self.N*2)
        
        # We want to calculate the difference between consecutive spectral bands
        diff_true = y_true[:, :, :, 1:] - y_true[:, :, :, :-1]
        diff_pred = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
        
        # Compute the difference of differences and square them
        diff_diff = diff_true - diff_pred
        slope_loss = torch.mean(diff_diff ** 2)  # Mean squared error over all pixels and bands
        
        if epoch == 0: 
            return self.mse_lambda*mse + self.gradient_lambda*slope_loss
        else:
            norm = self.mse_lambda + self.gradient_lambda * 0.1 **(epoch//10)
            lamd_slope = self.gradient_lambda * 0.1 ** (epoch // 10)
            total_loss = self.mse_lambda/norm * mse + lamd_slope/norm * slope_loss
            return total_loss

class gradientLoss(nn.Module):
    def __init__(self, N, mse_lambda = 2, gradient_lambda = 2e-1):
        super().__init__()
        self.mse_lambda = mse_lambda
        self.gradient_lambda = gradient_lambda
        self.N = N


    def forward(self, pred, gt, epoch=0):

        mse = F.mse_loss(pred, gt) / (self.N*2)

        pred_diff = torch.diff(pred, dim =  1)
        gt_diff = torch.diff(gt, dim = 1)

        pred_diff_flat = pred_diff.view(-1, pred_diff.shape[1])
        gt_diff_flat = gt_diff.view(-1, gt_diff.shape[1])

        cosine_sim = torch.cosine_similarity(pred_diff_flat, gt_diff_flat, dim = 1)
        slope_loss = 1-cosine_sim
        slope_loss = slope_loss.mean()/self.N

        if epoch == 0: 
            return self.mse_lambda*mse + self.gradient_lambda*slope_loss
        else:
            norm = self.mse_lambda + self.gradient_lambda * 0.1 **(epoch//10)
            lamd_slope = self.gradient_lambda * 0.1 ** (epoch // 10)
            total_loss = self.mse_lambda/norm * mse + lamd_slope/norm * slope_loss
            return total_loss



from scipy.spatial.distance import euclidean

def denoise_hyperspectral_image(image, threshold):
    """
    Denoise a hyperspectral image tensor by replacing noisy pixels with their closest neighbor.
    
    Args:
        image (torch.Tensor): 4D tensor representing the hyperspectral image (batch_size, bands, height, width)
        threshold (float): Threshold for considering a pixel as noise
        
    Returns:
        torch.Tensor: Denoised hyperspectral image tensor
    """
    batch_size, bands, height, width = image.shape
    
    # Create a copy of the image to hold the denoised output
    denoised_image = image.clone()
    
    # Iterate over the batch
    for b in range(batch_size):
        # Iterate over every pixel in the spatial dimensions
        for i in range(height):
            for j in range(width):
                # Get the current pixel (bands vector) at position (i, j)
                current_pixel = image[b, :, i, j]
                
                # Gather valid neighboring pixels
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connected neighborhood
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbors.append(image[b, :, ni, nj])
                
                # Stack neighbors into a tensor
                if neighbors:
                    neighbors = torch.stack(neighbors)  # Shape: (num_neighbors, bands)
                    
                    # Compute Euclidean distances between the current pixel and its neighbors
                    distances = torch.norm(neighbors - current_pixel, dim=1)
                    
                    # Calculate the average distance
                    avg_distance = distances.mean()
                    
                    # If the average distance is above the threshold, replace with the closest neighbor
                    if avg_distance > threshold:
                        closest_neighbor = neighbors[torch.argmin(distances)]
                        denoised_image[b, :, i, j] = closest_neighbor

    return denoised_image
### ====================================================================================================


# Training and validation loops
gap = 10
from torch.cuda.amp import autocast, GradScaler
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, args):
    model = model.to(device)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for lr, hr, sr in train_loader:
                lr, hr, sr = lr.to(device), hr.to(device), sr.to(device)
                assert not torch.isnan(lr).any(), "NaN detected in tensor"
                assert not torch.isnan(hr).any(), "NaN detected in tensor"
                assert not torch.isnan(sr).any(), "NaN detected in tensor"
                optimizer.zero_grad()

                with autocast():
                    outputs = model(lr, sr)
                    if args.use_gradient_loss:
                        lossfn = gradientLoss(batch_size, args.mse_lambda, args.slope_lambda)
                        loss = lossfn(outputs, hr, epoch)
                    elif args.use_gradient_mse:
                        lossfn = SpectralDifferenceLoss(batch_size, args.mse_lambda, args.slope_lambda)
                        loss = lossfn(outputs, hr, epoch)
                    else:
                        loss = criterion(outputs, hr)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.update() 
                running_loss += loss.item() if not math.isnan(loss.item()) else 0
                pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})
                pbar.update(1)
        log_training_loss(epoch, loss, experiment_dir=exp_dir)

        if epoch % gap == 0:
            validate(model, val_loader, criterion, device, epoch, args)  
            # torch.save(model, f'./checkpoints/{dataset}_{scale}_Epoch{epoch}_10_Aug2024.pth')
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, experiment_dir= exp_dir)
            

def validate(model, val_loader, criterion, device, epoch, args, ssim_required = False):
    model.eval()
    
    val_loss = 0.0
    PSNR_tot = 0
    if ssim_required:
        SSIM_tot = 0
    SAM_tot = 0
    best_psnr = 0
    best_sam = math.inf
    if ssim_required:
        best_ssim = 0

    SAM_times = 0
    PSNR_times = 0
    if ssim_required:
        SSIM_times = 0

    times = 0
    with torch.no_grad():
        for lr, hr, sr in val_loader:
            # if times >= 500:
            #     break
            times+=1
            lr, hr, sr = lr.to(device), hr.to(device), sr.to(device)
            assert not torch.isnan(lr).any(), "NaN detected in tensor"
            assert not torch.isnan(hr).any(), "NaN detected in tensor"
            assert not torch.isnan(sr).any(), "NaN detected in tensor"
            outputs = model(lr.clamp(0,1), sr.clamp(0,1))
            if args.use_denoising:
                outputs = denoise_hyperspectral_image(outputs, 0.5)
            
            if args.use_gradient_loss: 
                lossfn = gradientLoss(batch_size, args.mse_lambda, args.slope_lambda)
                loss = lossfn(outputs, hr, epoch)
            else:
                loss = criterion(outputs.clamp(0,1), hr.clamp(0,1))
            val_loss += loss.item()

            # if times == 0:
            #     print("shape of outputs and hr: ", outputs.shape, hr.shape)
                # times +=1 
            
            psnr_ = PSNR(np.clip(outputs.cpu().numpy(),0,1), np.clip(hr.cpu().numpy(), 0, 1))
            if psnr_ > best_psnr:
                best_psnr = psnr_

            if ssim_required:
                ssim_ = SSIM(np.clip(outputs.cpu().numpy(),0,1), np.clip(hr.cpu().numpy(), 0, 1))
                if ssim_ > best_ssim:
                    best_ssim = ssim_

            sam_ = SAM((outputs.cpu().numpy()), (hr.cpu().numpy()))
            if sam_ < best_sam:
                best_sam = sam_

            SAM_tot += sam_ if not math.isnan(sam_) else 0
            SAM_times += 1 if not math.isnan(sam_) else 0

            PSNR_tot += psnr_ if not math.isnan(psnr_) else 0
            PSNR_times += 1 if not math.isnan(psnr_) else 0

            if ssim_required:
                SSIM_tot += ssim_ if not math.isnan(ssim_) else 0
                SSIM_times += 1 if not math.isnan(ssim_) else 0

            
                


    loader_size = len(val_loader)
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    if ssim_required:
        print(f" PSNR: {PSNR_tot/PSNR_times} ; SAM: {SAM_tot/SAM_times} ; SSIM: {SSIM_tot/SSIM_times}")
        print(f"Best PSNR : {best_psnr}, bestSAM: {best_sam}, bestSSIM: {best_ssim}")
        metrics = {'epoch':epoch, 'psnr':PSNR_tot/PSNR_times, 'ssim':ssim_, 'sam':SAM_tot/SAM_times, 'loss': val_loss / len(val_loader)}
        log_validation_metrics(metrics, experiment_dir=exp_dir)
    else: 
        print(f" PSNR: {PSNR_tot/PSNR_times} ; SAM: {SAM_tot/SAM_times}")
        print(f"Best PSNR : {best_psnr}, bestSAM: {best_sam}")
        metrics = {'epoch':epoch, 'psnr':PSNR_tot/PSNR_times, 'sam':SAM_tot/SAM_times, 'loss': val_loss / len(val_loader)}
        log_validation_metrics(metrics, experiment_dir=exp_dir)
        
    
    # logging.info(f'Validation - PSNR: {PSNR_tot/times:.2f}, SSIM: {SSIM_tot/times:.2f} SAM: {SAM_tot/times:.2f}, Loss: {val_loss/len(val_loader)}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    # img = torch.from_numpy(np.ascontiguousarray(
    #     np.transpose(img, (2, 0, 1)))).float()

    
    # to range min_max
    min_val = img.min()
    max_val = img.max()
    
    if min_val == max_val:
        return torch.ones_like(img)

    eps = 1e-8
    img = (img-min_val)*(min_max[1] - min_max[0]) / (max_val - min_val) + min_max[0] + eps

    return torch.clip(torch.from_numpy(img),0,1)

# Main Trainer
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device==torch.device('cuda'):
        print("cuda activated")


    # Dataset
    dataset = "WashingtonDC"
    l_resolution = 32
    h_resolution = 64
    scale = int(h_resolution//l_resolution)

    # Argparsing
    parser = argparse.ArgumentParser()

    # parser.add_argument('-c', '--config', type=str, default=f'config/sr_sr3_{l_resolution}_{h_resolution}_{dataset}.json',
    #                     help='JSON file for configuration')
    # parser.add_argument('-debug', '-d', action='store_true')

    parser.add_argument('--weights_path', type=str, default='') # Use if inference is to be done live, while training the model ( Needs editing in the Code, Currently incompatible )
    parser.add_argument('--dataset', type=str, default='Pavia') # WashingtonDC, Pavia, or Chikusei
    parser.add_argument('--epochs', type=int, default = 1) # Number of epochs for training

    parser.add_argument('--use_eca', action='store_true') # Whether or not to use ECA (Efficient Channel Attention)
    parser.add_argument('--use_nonlocal', action='store_true') # Whether or not to use nonlocal conv blocks
    parser.add_argument('--use_denoising', action='store_true')
    parser.add_argument('--use_gradient_loss', action='store_true')         
    parser.add_argument('--use_gradient_mse', action='store_true')
    parser.add_argument('--use_rcab', action='store_true')
    parser.add_argument('--use_spefe', action='store_true')
    parser.add_argument('--use_dilated_conv', action='store_true')
    parser.add_argument('--use_custom_multiscale', action='store_true')



    parser.add_argument('--experiment_name', type=str, default='GMOD_experiment') # Whether or not to use nonlocal conv blocks
    parser.add_argument('--mse_lambda', type=float, default=7e-1)
    parser.add_argument('--slope_lambda', type = float, default = 3e-1)

    args = parser.parse_args()

    dataset = args.dataset
    
    dataroot = r"D:/Tarun_P/Datasets"
    dataroot = f"{dataroot}/{dataset}/LR_HR/SR_{l_resolution}_{h_resolution}_{scale}x"
    # sr3_weights_path = r"D:\Tarun_P\SR3_HSI\SR3_HSI\experiments\CHIKUSEI_2X\checkpoint\I100000_E142_gen.pth"
    
    if dataset=='Chikusei':
        num_bands = 128
    elif dataset=='WashingtonDC':
        num_bands = 191
    elif dataset=='Pavia':
        num_bands = 102
    else:
        num_bands = 3
        
    print('num_bands: ', num_bands)

    upscale_factor = int(h_resolution//l_resolution)
    batch_size = 4
    num_epochs = args.epochs
    learning_rate = 1e-5

    # Dataset and DataLoader
    transform = ToTensor()
    train_dataset = GMOD_Dataset(dataroot, l_resolution, h_resolution)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
    val_dataset = GMOD_Dataset(dataroot, l_resolution, h_resolution, split='val')
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4)

    args = parser.parse_args()
    # opt = Logger.parse(args)

    # args.use_custom_multiscale = True
    # args.use_spefe = True
    # Model, criterion, optimizer
    model = GMOD(num_bands, upscale_factor,args, use_ECA=args.use_eca, use_nonlocal=args.use_nonlocal)
    criterion = nn.MSELoss()
    # criterion  = customLoss(num_bands)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(" Model Params : ", count_parameters(model))



    model = GMOD(num_bands, upscale_factor, 1, '')
    loaded_model = torch.load(r"D:\Tarun_P\GMOD2\GMOD_experiments\WashingtonFinal_20240929_060937\checkpoints\model_epoch_50.pth")

    # state_dict = loaded_model.state_dict()
    model.load_state_dict(loaded_model['model_state_dict'])

    lr = transform2tensor(np.load(r"D:\Tarun_P\Datasets\WashingtonDC\LR_HR\SR_32_64_2x\train\LR_32_64_train\washington_LR_train_30.npy").astype(np.float32)).unsqueeze(0)
    sr = transform2tensor(np.load(r"D:\Tarun_P\Datasets\WashingtonDC\PCA_LR_HR\SR_32_64_2x\train\SR3_32_64_train\SR3_30.npy").astype(np.float32)).unsqueeze(0)
    out = model(lr, sr)
    np.save("./outputs/gmod_washington30.npy", out.detach().cpu().numpy())

    validate(model, train_loader, criterion, torch.device('cuda'))
    params = sum(p.numel() for p in model.parameters())
    print(params)


    # Training
    exp_dir = create_experiment_folder(args.experiment_name)
    log_model_params(model, exp_dir)
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, args)
    save_checkpoint(model=model, optimizer=optimizer, epoch=num_epochs, experiment_dir= exp_dir)
    # validate(model, val_loader, criterion, device, epoch, ssim_required = False)
    validate(model=model, val_loader=val_loader, criterion=criterion, device=device, epoch=num_epochs, args=args, ssim_required=True)

    

    # torch.save(model, f'./checkpoints/{dataset}_{scale}_Epoch{args.epochs}_10_Aug2024.pth')
    # test(model)
