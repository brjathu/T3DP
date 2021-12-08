from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import math
import numbers
import itertools

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import cv2

import copy
import logging
from matplotlib import gridspec
import torch 
    
from PIL import Image
import random

class ReIDLoss(nn.Module):
    
    def __init__(self, cosine=True):
        super(ReIDLoss, self).__init__()
        self.L2_loss = nn.MSELoss()
        self.cosine  = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.use_cosine = cosine
        
    def forward(self, ids_x, embeddings, T, P):
        
        c = 0.0
        # prediction loss for delta_2
        re_id_loss = torch.tensor(0).float().cuda()
        unique_id_ = np.unique(ids_x)
        unique_ids = []
        for u_ in unique_id_:
            if(u_!=-1):
                unique_ids.append(u_)
                

        for idx in unique_ids:
            if(idx!=-1):
                
                p_distance = 0
                n_distance = 0
                
                # positives
                loc_ = np.where(ids_x==idx)[0]   
                random.shuffle(loc_)
                same_embeddings = embeddings[loc_]
                if(len(loc_)>=2):
                    
                    if(self.use_cosine):
                        p_distance   = 1 - self.cosine(same_embeddings[0:1], same_embeddings[1:2])
                        p_distance   = torch.clamp(p_distance-0.0, min=0.0)
                    else:
                        p_distance   = F.pairwise_distance(same_embeddings[0:1], same_embeddings[1:2], keepdim = True)
                    
                # negatives
                loc_n = []
                try:
                    loc_n = np.where(unique_ids!=idx)[0] 
                    random.shuffle(loc_n)
                    negative_id = unique_ids[loc_n[0]]
                    loc_n = np.where(ids_x==negative_id)[0]   
                    random.shuffle(loc_n)
                    negative_embeddings = embeddings[loc_n] 
                    if(len(loc_n)>=1):
                        if(self.use_cosine):
                            n_distance     = 1 - self.cosine(same_embeddings[0:1], negative_embeddings[0:1])
                            n_distance     = torch.clamp(2.0-n_distance, min=0.0)
                        else:
                            n_distance     = F.pairwise_distance(same_embeddings[0:1], negative_embeddings[0:1], keepdim = True)
                            n_distance     = torch.pow(torch.clamp(100.0 - n_distance, min=0.0), 2)
                except:
                    pass

                if(len(loc_)>=2 ):
                    re_id_loss += torch.mean(p_distance)
                if(len(loc_n)>=1 ):
                    re_id_loss += torch.mean(n_distance)

                c +=1

        return re_id_loss/c
