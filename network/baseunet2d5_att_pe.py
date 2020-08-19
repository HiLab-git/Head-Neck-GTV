# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pymic.layer.activation import get_acti_func
from pymic.layer.convolution import ConvolutionLayer
from pymic.layer.deconvolution import DeconvolutionLayer


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
            kernel_size, paddding, acti_func, acti_func_param):
        super(UNetBlock, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        self.conv1 = ConvolutionLayer(in_channels,  out_channels, kernel_size = kernel_size, 
                padding = paddding, acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = ConvolutionLayer(out_channels, out_channels, kernel_size = kernel_size, 
                padding = paddding, acti_func=get_acti_func(acti_func, acti_func_param))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()

        
        self.in_chns = in_channels
        self.out_chns = out_channels
        self.acti_func1 = nn.ReLU()
        self.acti_func2 = nn.Sigmoid()
        self.fc1 = nn.Conv3d(self.in_chns, self.out_chns, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.fc2 = nn.Conv3d(self.out_chns, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        
    def forward(self, x):
        f1 = self.fc1(x)
        f1 = self.acti_func1(f1)
        f2 = self.fc2(f1)
        f2_a = self.acti_func2(f2)
        return f2_a*x + x , f2_a

class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor


class Baseunet2d5_att_pe(nn.Module):
    def __init__(self, params):
        super(Baseunet2d5_att_pe, self).__init__()
        self.params = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.n_class   = self.params['class_num']
        self.acti_func = self.params['acti_func']
        self.dropout   = self.params['dropout']
        assert(len(self.ft_chns) == 5)

        self.block1 = UNetBlock(self.in_chns, self.ft_chns[0], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block2 = UNetBlock(self.ft_chns[0], self.ft_chns[1], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block3 = UNetBlock(self.ft_chns[1], self.ft_chns[2], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block4 = UNetBlock(self.ft_chns[2], self.ft_chns[3], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block5 = UNetBlock(self.ft_chns[3], self.ft_chns[4], 
            (3, 3, 3), (1, 1, 1), self.acti_func, self.params)

        self.block6 = UNetBlock(self.ft_chns[3] * 2, self.ft_chns[3], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block7 = UNetBlock(self.ft_chns[2] * 2, self.ft_chns[2], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block8 = UNetBlock(self.ft_chns[1] * 2, self.ft_chns[1], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block9 = UNetBlock(self.ft_chns[0] * 2, self.ft_chns[0], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.down1 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))
        self.down2 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))
        self.down3 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))
        self.down4 = nn.MaxPool3d(kernel_size = 2)

        self.up1 = DeconvolutionLayer(self.ft_chns[4], self.ft_chns[3], kernel_size = 1,
            stride = 1, acti_func = get_acti_func(self.acti_func, self.params))
        self.up2 = DeconvolutionLayer(self.ft_chns[3], self.ft_chns[2], kernel_size = (1, 2, 2),
            stride = (1, 2, 2), acti_func = get_acti_func(self.acti_func, self.params))
        self.up3 = DeconvolutionLayer(self.ft_chns[2], self.ft_chns[1], kernel_size = (1, 2, 2),
            stride = (1, 2, 2), acti_func = get_acti_func(self.acti_func, self.params))
        self.up4 = DeconvolutionLayer(self.ft_chns[1], self.ft_chns[0], kernel_size = (1, 2, 2),
            stride = (1, 2 ,2), acti_func = get_acti_func(self.acti_func, self.params))

        self.conv = nn.Conv3d(self.ft_chns[0], self.n_class, 
            kernel_size = (1, 3, 3), padding = (0, 1, 1))

        
        self.res1 = UNetBlock(self.in_chns, self.ft_chns[0], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.res2 = UNetBlock(self.ft_chns[0], self.ft_chns[1], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.res3 = UNetBlock(self.ft_chns[1], self.ft_chns[2], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.res4 = UNetBlock(self.ft_chns[2], self.ft_chns[3], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.res5 = UNetBlock(self.ft_chns[3], self.ft_chns[4], 
            (3, 3, 3), (1, 1, 1), self.acti_func, self.params)

        self.res6 = UNetBlock(self.ft_chns[3] * 2, self.ft_chns[3], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.res7 = UNetBlock(self.ft_chns[2] * 2, self.ft_chns[2], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.res8 = UNetBlock(self.ft_chns[1] * 2, self.ft_chns[1], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.res9 = UNetBlock(self.ft_chns[0] * 2, self.ft_chns[0], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)


        self.pe1 = ProjectExciteLayer(self.ft_chns[0])
        self.pe2 = ProjectExciteLayer(self.ft_chns[1])
        self.pe3 = ProjectExciteLayer(self.ft_chns[2])
        self.pe4 = ProjectExciteLayer(self.ft_chns[3])
        self.pe5 = ProjectExciteLayer(self.ft_chns[3] * 2)
        self.pe6 = ProjectExciteLayer(self.ft_chns[2] * 2)
        self.pe7 = ProjectExciteLayer(self.ft_chns[1] * 2)
        self.pe8 = ProjectExciteLayer(self.ft_chns[0] * 2)

        # dropout
        if(self.dropout):
            self.drop1 = nn.Dropout(p=0)
            self.drop2 = nn.Dropout(p=0)
            self.drop3 = nn.Dropout(p=0)
            self.drop4 = nn.Dropout(p=0.3)
            self.drop5 = nn.Dropout(p=0.5)
        

        #attention block(two gates)
        self.attention1 = AttentionBlock(self.ft_chns[3] * 2 , self.ft_chns[3])
        self.attention2 = AttentionBlock(self.ft_chns[2] * 2, self.ft_chns[2])
        self.attention3 = AttentionBlock(self.ft_chns[1] * 2, self.ft_chns[1])
        self.attention4 = AttentionBlock(self.ft_chns[0] * 2, self.ft_chns[0])
        self.attention5 = AttentionBlock(self.ft_chns[3], self.ft_chns[3]//2)

    def forward(self, x):

        f1 = self.block1(x)
        if(self.dropout):
             f1 = self.drop1(f1)
        d1 = self.down1(f1)
        d1 = self.pe1(d1)
        f2 = self.block2(d1)
        if(self.dropout):
             f2 = self.drop1(f2)
        d2 = self.down2(f2)
        d2 = self.pe2(d2)
        f3 = self.block3(d2)
        if(self.dropout):
             f3 = self.drop1(f3)
        d3 = self.down3(f3)
        d3 = self.pe3(d3)
        f4 = self.block4(d3)
        if(self.dropout):
             f4 = self.drop1(f4)
        d4 = f4
        d4, att5 = self.attention5(d4)
        d4 = self.pe4(d4)
        f5 = self.block5(d4)
        if(self.dropout):
             f5 = self.drop1(f5)


        
        f5up  = self.up1(f5)
        f4cat = torch.cat((f4, f5up), dim = 1)
        f4cat, att1 = self.attention1(f4cat)
        f4cat = self.pe5(f4cat)
        # f4cat = self.se4(f4cat)
        f6    = self.block6(f4cat)
        

        f6up  = self.up2(f6)
        f3cat = torch.cat((f3, f6up), dim = 1)
        f3cat, att2= self.attention2(f3cat)
        f3cat = self.pe6(f3cat)
        # f3cat = self.se3(f3cat)
        f7    = self.block7(f3cat)

        f7up  = self.up3(f7)
        f2cat = torch.cat((f2, f7up), dim = 1)
        f2cat, att3 = self.attention3(f2cat)
        f2cat = self.pe7(f2cat)
        f8    = self.block8(f2cat)
        

        f8up  = self.up4(f8)
        f1cat = torch.cat((f1, f8up), dim = 1)
        f1cat, att4 = self.attention4(f1cat)
        f1cat = self.pe8(f1cat)
        f9    = self.block9(f1cat)
       

        output = self.conv(f9)
        return output
        # return output

if __name__ == "__main__":
    params = {'in_chns':1,
              'feature_chns':[8, 16, 32, 64, 128],
              'class_num': 2,
              'acti_func': 'leakyrelu',
              'leakyrelu_alpha': 0.01,
              'is_batchnorm': True,
              'dropout': True,
              'nonlocal_mode':'concatenation'}
    Net = Baseunet2d5_att_pe(params)
    Net = Net.double()

    x  = np.random.rand(4, 1, 32, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    y = y.detach().numpy()
    print(y.shape)
