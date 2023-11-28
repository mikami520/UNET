'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2023-11-28 13:49:29
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2023-11-28 14:27:54
FilePath: /UNET/train.py
Description: 
I Love IU
Copyright (c) 2023 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
import monai
import torch
import torch.nn as nn
from monai.networks.nets import UNet
import numpy as np
