import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import BatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm
from .import net_blocks as nb

class EncodingHead(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, img_H=64, img_W=128):
        super(EncodingHead, self).__init__()
        
        #Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1, stride=2)
        self.conv5_1 = nn.Conv2d(256, 8, 3, padding=1, stride=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        
        
        #Decoder
        self.t_conv5_1 = nn.Conv2d(8, 256, 3, stride=1,  padding=1)
        self.t_conv5 = nn.Conv2d(256, 256, 3, stride=1,  padding=1)
        self.t_conv4 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.t_conv3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.t_conv2 = nn.Conv2d(64, 16, 3, stride=1, padding=1)
        self.t_conv1 = nn.Conv2d(16, 3, 3, stride=1, padding=1)
        self.t_conv1_1 = nn.Conv2d(3, 3, 3, stride=1, padding=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))   
        x = F.relu(self.conv5_1(x))  
        
        
        feat = x.view(x.size(0), -1)
        
        
        x = F.relu(self.t_conv5_1(x))
        
        x = F.relu(self.t_conv5(x))
        x = self.up(x)
        
        x = F.relu(self.t_conv4(x))
        x = self.up(x)
        
        x = F.relu(self.t_conv3(x))
        x = self.up(x)
        
        x = F.relu(self.t_conv2(x))
        x = self.up(x)
        
        x = F.relu(self.t_conv1(x))
        x = self.up(x)
        
        x = F.tanh(self.t_conv1_1(x))
              
        return x, feat
