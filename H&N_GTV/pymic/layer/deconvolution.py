# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn

class DeconvolutionLayer(nn.Module):
    """
    A compose layer with the following components:
    deconvolution -> (batch_norm) -> activation -> (dropout)
    batch norm and dropout are optional
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
            dim = 3, stride = 1, padding = 0, output_padding = 0, 
            dilation =1, groups = 1, bias = True, 
            batch_norm = True, acti_func = None):
        super(DeconvolutionLayer, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.batch_norm = batch_norm
        self.acti_func  = acti_func
        
        assert(dim == 2 or dim == 3)
        if(dim == 2):
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size, stride, padding, output_padding,
                groups, bias, dilation)
            if(self.batch_norm):
                self.bn = nn.modules.BatchNorm2d(out_channels)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                kernel_size, stride, padding, output_padding,
                groups, bias, dilation)
            if(self.batch_norm):
                self.bn = nn.modules.BatchNorm3d(out_channels)

    def forward(self, x):
        f = self.conv(x)
        if(self.batch_norm):
            f = self.bn(f)
        if(self.acti_func is not None):
            f = self.acti_func(f)
        return f
