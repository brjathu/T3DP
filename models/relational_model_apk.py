# Code adapted from: https://github.com/lucidrains/vit-pytorch

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim_all, fn):
        super().__init__()
        self.dim_all = dim_all # [dim, dim_a, dim_p, dim_k]
        self.norm_a  = nn.LayerNorm(dim_all[1])
        self.norm_p  = nn.LayerNorm(dim_all[2])
        self.norm_k  = nn.LayerNorm(dim_all[3])
        self.fn      = fn
    def forward(self, x, **kwargs):
        x_a = self.norm_a(x[:, :, :self.dim_all[1]])
        x_p = self.norm_p(x[:, :, self.dim_all[1]:self.dim_all[1]+self.dim_all[2]])
        x_k = self.norm_k(x[:, :, self.dim_all[1]+self.dim_all[2]:])
        x   = torch.cat([x_a, x_p, x_k], -1)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim_all, hidden_dim, dropout = 0.):
        super().__init__()
        self.dim_all   = dim_all
        hidden_dim_a   = dim_all[1]
        hidden_dim_p   = dim_all[2]
        hidden_dim_k   = dim_all[3]
        self.net_a     = nn.Sequential(nn.Linear(self.dim_all[1], hidden_dim_a), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim_a, self.dim_all[1]), nn.Dropout(dropout))
        self.net_p     = nn.Sequential(nn.Linear(self.dim_all[2], hidden_dim_p), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim_p, self.dim_all[2]), nn.Dropout(dropout))
        self.net_k     = nn.Sequential(nn.Linear(self.dim_all[3], hidden_dim_k), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim_k, self.dim_all[3]), nn.Dropout(dropout))
        
        
    def forward(self, x):
        x_a = self.net_a(x[:, :, :self.dim_all[1]])
        x_p = self.net_p(x[:, :, self.dim_all[1]:self.dim_all[1]+self.dim_all[2]])
        x_k = self.net_k(x[:, :, self.dim_all[1]+self.dim_all[2]:])
        x   = torch.cat([x_a, x_p, x_k], -1)
        return x

class Attention(nn.Module):
    def __init__(self, dim_all, heads = 8, dim_head = 64, dropout = 0., betas=[1,1,1]):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim_all[0])
        
        self.beta1      = torch.nn.Parameter(torch.rand(1,)*0.0 + 1.0/3)
        self.beta2      = torch.nn.Parameter(torch.rand(1,)*0.0 + 1.0/3)
        self.beta3      = torch.nn.Parameter(torch.rand(1,)*0.0 + 1.0/3)
        
        self.heads      = heads
        self.scale      = dim_head ** -0.5
        
        self.dim_a      = dim_all[1]
        self.dim_p      = dim_all[2]
        self.dim_k      = dim_all[3]
        self.dim_ah     = self.dim_a//heads
        self.dim_ph     = self.dim_p//heads
        self.dim_kh     = self.dim_k//heads
        
        self.Wqkv_a = nn.Linear(self.dim_a, self.dim_ah*3, bias = False)
        self.Wqkv_p = nn.Linear(self.dim_p, self.dim_ph*3, bias = False)
        self.Wqkv_k = nn.Linear(self.dim_k, self.dim_kh*3, bias = False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim_all[0]), nn.Dropout(dropout)) if project_out else nn.Identity()

        self.dropout = nn.Dropout(p=dropout)
        self.betas   = np.array(betas)/np.sum(betas)
        
    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        
        # x     -> (16, 100, 2610)
        qkv_a = self.Wqkv_a(x[:, :, :self.dim_a]).chunk(3, dim = -1)                         # qkv_a -> (16, 100, 512*3)
        qkv_p = self.Wqkv_p(x[:, :, self.dim_a:self.dim_a+self.dim_p]).chunk(3, dim = -1)    # qkv_p -> (16, 100, 2048*3)
        qkv_k = self.Wqkv_k(x[:, :, self.dim_a+self.dim_p:]).chunk(3, dim = -1)              # qkv_k -> (16, 100, 30*3)
        
        
        q_a, k_a, v_a = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv_a)    # q_a   -> (16, 1, 100, 512)
        q_p, k_p, v_p = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv_p)    # q_p   -> (16, 1, 100, 2048)
        q_k, k_k, v_k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv_k)    # q_k   -> (16, 1, 100, 30)
        
        v = torch.cat([v_a, v_p, v_k], -1)   # v   -> (16, 1, 100, 2048)
        # q = torch.cat([q_a, q_p, q_k], -1)   # q   -> (16, 1, 100, 2048)
        # k = torch.cat([k_a, k_p, k_k], -1)   # k   -> (16, 1, 100, 2048)
        # # average attention  dots-> (16, 1, 100, 100)
        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
    
    
    
        q = torch.cat([q_a], -1)   # q   -> (16, 1, 100, 2048)
        k = torch.cat([k_a], -1)   # k   -> (16, 1, 100, 2048)
        dots_a = einsum('b h i d, b h j d -> b h i j', q, k) * (self.dim_a**-0.5)
        
        
        q = torch.cat([q_p], -1)   # q   -> (16, 1, 100, 2048)
        k = torch.cat([k_p], -1)   # k   -> (16, 1, 100, 2048)
        dots_p = einsum('b h i d, b h j d -> b h i j', q, k) * (self.dim_p**-0.5)
        

        q = torch.cat([q_k], -1)   # q   -> (16, 1, 100, 2048)
        k = torch.cat([k_k], -1)   # k   -> (16, 1, 100, 2048)
        dots_k = einsum('b h i d, b h j d -> b h i j', q, k) * (self.dim_k**-0.5)

        # dots = self.beta1 * dots_a + self.beta2 * dots_p + self.beta3 * dots_k
        if(len(mask)==2):
            mask_0 = mask[0]>0
            assert mask_0.shape[-1] == dots_a.shape[-1], 'mask_0 has incorrect dimensions'
            assert mask_0.shape[-1] == dots_p.shape[-1], 'mask_0 has incorrect dimensions'
            assert mask_0.shape[-1] == dots_k.shape[-1], 'mask_0 has incorrect dimensions'
            mask_0 = rearrange(mask_0, 'b i -> b () i ()') * rearrange(mask_0, 'b j -> b () () j')
            
            mask_1 = mask[1]>0
            mask_1 = mask_1.unsqueeze(0).unsqueeze(0)
            mask_1 = mask_1.repeat(b, 1, 1, 1)  
            mask_ = torch.logical_and(mask_0, mask_1)

        else:
            mask_ = mask>0
        if mask_ is not None:
            dots_a.masked_fill_(~mask_, -1e10)
            dots_p.masked_fill_(~mask_, -1e10)
            dots_k.masked_fill_(~mask_, -1e10)
            del mask_

            

        dots_a = dots_a.softmax(dim=-1)
        dots_p = dots_p.softmax(dim=-1)
        dots_k = dots_k.softmax(dim=-1)
        
        attn = self.betas[0] * dots_a + self.betas[1] * dots_p + self.betas[2] * dots_k
        attn = self.dropout(attn)
        
        
        # out -> (16, 1, 100, 2610)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # out -> (16, 100, 2610)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class RelationTransformerModel_APK(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., betas = [1,1,1]):
        super().__init__()
        self.layers   = nn.ModuleList([])
        dim_a         = dim[1]
        dim_p         = dim[2]
        dim_k         = dim[3]
        dim_all       = dim
        self.dim_all  = dim_all
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim_all, Attention(dim_all, heads = heads, dim_head = dim_head, dropout = dropout, betas=betas))),
                Residual(PreNorm(dim_all, FeedForward(dim_all, mlp_dim, dropout = dropout)))
            ]))
            
        for layer in self.layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
            
        return x
