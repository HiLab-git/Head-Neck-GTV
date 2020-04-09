# -*- coding: utf-8 -*-
from __future__ import print_function, division

from pymic.net3d.unet2d5 import UNet2D5
from pymic.net3d.unet3d import UNet3D
from pymic.net3d.baseunet2d5_att_pe import Baseunet2d5_att_pe
from pymic.net3d.baseunet2d5_pe import UNet2D5_PE
from pymic.net3d.vnet import VNet

def get_network(params):
    net_type = params['net_type']
    if(net_type == 'UNet2D5'):
        return UNet2D5(params)
    if(net_type == 'Baseunet2d5_att_pe'):
        return Baseunet2d5_att_pe(params)
    if(net_type == 'VNet'):
        return VNet()
    if(net_type == 'UNet2D5_SE'):
        return UNet2D5_SE(params)    
    elif(net_type == 'UNet3D'):
        return UNet3D(params)
    else:
        raise ValueError("undefined network {0:}".format(net_type))
