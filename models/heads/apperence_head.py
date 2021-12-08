import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import BatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm

from .import net_blocks as nb

class TextureHead(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, nz_feat, uv_sampler, opts, img_H=64, img_W=128, n_upconv=6, nc_init=32, predict_flow=False, symmetric=False, num_sym_faces=624):
        super(TextureHead, self).__init__()
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init

        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)

        nc_final=2
        
        self.decoder = decoder(n_upconv, nc_init, nc_final, self.feat_H, self.feat_W)

    def forward(self, feat, skips):
        
        flow = self.decoder.forward(feat, skips)
        flow = torch.nn.functional.tanh(flow)
        
        return flow
    

class decoder(nn.Module):
    def __init__(self, n_upconv, nc_init, nc_final, H, W):
        super().__init__()
        
        self.n_upconv = n_upconv
        self.nc_init  = nc_init
        self.nc_final = nc_final
        self.feat_H = H
        self.feat_W = W
        nf = 8
        
        self.G_middle_0 = ResnetBlock(2048, 8 * nf)
        self.G_middle_1 = ResnetBlock(8 * nf, 8 * nf)

        self.G_middle_2 = ResnetBlock(1024, 8 * nf)
        self.G_middle_2_1 = ResnetBlock(2*8 * nf, 8 * nf)
        
        self.G_middle_3 = ResnetBlock(512, 8 * nf)
        self.G_middle_3_1 = ResnetBlock(2*8 * nf, 8 * nf)
        
        self.G_middle_4 = ResnetBlock(256, 4 * nf)
        self.G_middle_4_1 = ResnetBlock(2*4 * nf, 4 * nf)
        
        
        self.up_0 = ResnetBlock(8 * nf, 4 * nf)
        self.up_1 = ResnetBlock(4 * nf, 4 * nf)
        self.up_2 = ResnetBlock(4 * nf, 2 * nf)
        self.up_3 = ResnetBlock(2 * nf, 1 * nf)

        self.conv_img = nn.Conv2d(nf, self.nc_final, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        
        
        
    def forward(self, conv_featz, skips):
        x1,x2,x3,x4 = skips
        
        x = conv_featz
        x = self.G_middle_0(x)
        if self.n_upconv >= 6:
            x = self.up(x)

        x = torch.cat([x, self.G_middle_2(x3)], 1)
        x = self.G_middle_2_1(x)
        x = self.G_middle_1(x)
        
        x = self.up(x)
        x = torch.cat([x, self.G_middle_3(x2)], 1)
        x = self.G_middle_3_1(x)
        x = self.up_0(x)
        
        x = self.up(x)
        x = torch.cat([x, self.G_middle_4(x1)], 1)
        x = self.G_middle_4_1(x)
        x = self.up_1(x)
        
        x = self.up(x)
        x = self.up_2(x)
        x = self.up(x)
        x = self.up_3(x)

        if self.n_upconv >= 7:
            x = self.up(x)
            x = self.up_4(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x



class ResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        # if opts.SPADE_spectral_norm:
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)
            
            

        # define normalization layers
        self.norm_0 = BatchNorm2d(fin)
        self.norm_1 = BatchNorm2d(fmiddle)
        if self.learned_shortcut:
            self.norm_s = BatchNorm2d(fin)

            
    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x):
        # seg is texture structure
        x_s = self.shortcut(x)
        dx = self.conv_0(self.actvn(self.norm_0(x)))
        dx = self.conv_1(self.actvn(self.norm_1(dx)))
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)