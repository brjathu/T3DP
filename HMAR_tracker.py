import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import os
import pyrender
import trimesh
import warnings


import cv2
import math
import time
import numpy as np
import wandb
import random
import copy
import argparse
from tqdm import tqdm


from models.hmar import HMAR
from models.relational_model_apk import RelationTransformerModel_APK


     
def positionalencoding1d(d_model, length):
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class keypoint_encoder(nn.Module):
    def __init__(self, embedding_size_in, embedding_size_out):
        super(keypoint_encoder, self).__init__()
        self.layer1 = nn.Linear(embedding_size_in, embedding_size_out)
        
    def forward(self, keypoints):
        x_ = self.layer1(keypoints)
        return x_
    
    
class HMAR_tracker(nn.Module):

    def __init__(self, mode='A', betas=[1.0,1.0,1.0]):
        super(HMAR_tracker, self).__init__()
        
        self.device=torch.device('cuda')
        self.mode = mode

        self.A_size = 512
        self.P_size = 2048
        self.K_size = 15*4
        self.betas  = betas
        
        
        self.keypoint_encoder     = keypoint_encoder(self.K_size, self.K_size)
        
        #######################  Single Attibute Model ##############################################################
        self.total_size           = self.A_size + self.P_size + self.K_size*2 
        self.relation_transformer = RelationTransformerModel_APK([self.total_size, self.A_size, self.P_size, self.K_size*2], 
                                                                 depth    = 1, 
                                                                 heads    = 1,
                                                                 dim_head = self.total_size, 
                                                                 mlp_dim  = self.total_size,
                                                                 dropout  = 0.,
                                                                 betas    = self.betas)
        ##############################################################################################################

    def forward(self, BS, T, P, embeddings, ids, bboxs, keypoints):
        
        if ("K" in self.mode):
            temporal_embedding      = positionalencoding1d(self.K_size, T)
            temporal_embedding      = temporal_embedding.unsqueeze(0)        
            temporal_embedding      = temporal_embedding.unsqueeze(2)         
            temporal_embedding      = temporal_embedding.repeat(BS, 1, P, 1)  
            temporal_embedding      = temporal_embedding.cuda()
        
            keypoints_3d            = torch.cat((keypoints[:, :, :, :, 0], keypoints[:, :, :, :, 1], keypoints[:, :, :, :, 2], keypoints[:, :, :, :, 2]), -1)     
            keypoints_3d            = keypoints_3d.cuda()
            keypoints_3d            = keypoints_3d.view(BS*T*P, -1)          
                
            embedding_key_3d        = self.keypoint_encoder(keypoints_3d)        
            embedding_key_3d        = embedding_key_3d.view(BS, T, P, self.K_size)
            keypoints_3d            = keypoints_3d.view(BS, T, P, self.K_size)
            embedding_key_3d        = torch.cat((embedding_key_3d, temporal_embedding), 3) 
            
        
        ids_x                       = ids.view(BS,T*P)
        mask_ids                    = torch.where(ids_x==-1)
        mask_x                      = torch.ones_like(ids_x)
        mask_x[mask_ids]            = 0.0
        mask_x                      = mask_x.cuda()

        mask_a                      = torch.zeros((T*P, T*P))
        mask_a[:, :]                = 1.0
        mask_a                      = mask_a.cuda()         
             
                    
        
        input_embeddings            = torch.cat((embeddings[1]*self.betas[0], embeddings[0]*self.betas[1], embedding_key_3d*self.betas[2]), 3) 
        input_embeddings            = input_embeddings.view(BS, T*P, -1)
        output_embeddings           = self.relation_transformer(input_embeddings, [mask_x, mask_a])
        output_embeddings           = torch.cat((output_embeddings[:, :, :self.A_size]*self.betas[0], 
                                                 output_embeddings[:, :, self.A_size:self.A_size+self.P_size].view(BS, T*P, -1)*self.betas[1], 
                                                 output_embeddings[:, :, self.A_size+self.P_size:].view(BS, T*P, -1)*self.betas[2]), -1) 
        
        output                      = {}
        output["output_embeddings"] = output_embeddings
        output["ids"]               = ids_x
        output["mask"]              = mask_x

        
        return output, 0
        
        
    def forward_edge_loss(self, emb1, emb2, target):
        output = self.edge_classifier(emb1, emb2)
        return F.binary_cross_entropy_with_logits(output, target.view(-1, 1))
        
        
    def normalize_embeddings(self, x):
        norm = x.norm(p=2, dim=-1, keepdim=True)
        x_normalized = x.div(norm.expand_as(x))
        return x_normalized
     
