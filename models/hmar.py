import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import os
import json
import copy
import heapq
from PIL import Image
from tqdm import tqdm
import argparse
import cv2
import numpy as np

from .backbones import resnet
from .heads.mesh import *
from .heads.smpl_head import SMPLHead
from .heads.apperence_head import TextureHead
from .heads.encoding_head import EncodingHead
from .joint_mapper import JointMapper, smpl_to_openpose
from .smplx import create
from .utils import perspective_projection

from yacs.config import CfgNode as CN
from .renderer import Renderer
from .utils import *

class HMAR(nn.Module):
    
    def __init__(self, config):
        super(HMAR, self).__init__()
        with open(config, 'r') as f:
            cfg = CN.load_cfg(f); cfg.freeze()
        self.cfg = cfg

        nz_feat  = 512
        tex_size = 6
        img_H    = 256
        img_W    = 256
            
        texture_file         = np.load(self.cfg.SMPL.TEXTURE)
        self.faces_cpu       = texture_file['smpl_faces'].astype('uint32')
        
        # NMR init
        vt                   = texture_file['vt']
        ft                   = texture_file['ft']
        uv_sampler           = compute_uvsampler(vt, ft, tex_size=6)
        uv_sampler           = torch.tensor(uv_sampler, dtype=torch.float)
        uv_sampler           = uv_sampler.unsqueeze(0)

        self.F               = uv_sampler.size(1)   
        self.T               = uv_sampler.size(2)
        self.uv_sampler      = uv_sampler.view(-1, self.F, self.T*self.T, 2)
        self.backbone        = resnet(cfg.MODEL.BACKBONE, num_layers=self.cfg.MODEL.BACKBONE.NUM_LAYERS, pretrained=True)
        self.texture_head    = TextureHead(nz_feat, self.uv_sampler, self.cfg, img_H=img_H, img_W=img_W)
        self.encoding_head   = EncodingHead(img_H=img_H, img_W=img_W)
    
        smpl_params         = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        joint_mapper         = JointMapper(smpl_to_openpose(model_type=cfg.SMPL.MODEL_TYPE))
        self.smpl           = create(**smpl_params,
                                  joint_mapper = joint_mapper,
                                  create_betas=False,
                                  create_body_pose=False,
                                  create_global_orient=False,
                                  create_left_hand_pose=False,
                                  create_right_hand_pose=False,
                                  create_expression=False,
                                  create_leye_pose=False,
                                  create_reye_pose=False,
                                  create_jaw_pose=False,
                                  create_transl=False)
        
        self.neural_renderer = Renderer(focal_length=self.cfg.EXTRA.FOCAL_LENGTH, img_res=256, faces=self.faces_cpu)
        
        self.smpl_head      = SMPLHead(cfg)
        self.smpl_head.pool = 'pooled'
        
    def forward(self, x):
        feats, skips = self.backbone(x)
        flow = self.texture_head(feats, skips)
        uv_image, texture_embedding = self.encode_texture(flow, x)
        
        pose_embeddings = feats.max(3)[0].max(2)[0]
        pose_embeddings = pose_embeddings.view(x.size(0),-1)
        texture_embedding = texture_embedding.view(x.size(0), -1)
        
        return pose_embeddings, texture_embedding, flow, uv_image
    
    def forward_for_smpl(self, pose_embeddings):
        with torch.no_grad():
            pred_smpl_params, pred_cam = self.smpl_head(pose_embeddings)
        
        return pred_smpl_params
    
    def encode_texture(self, flow_map, img_x):
        batch_size = flow_map.size(0)
        flow_map = flow_map.permute(0,2,3,1)
        uv_images = torch.nn.functional.grid_sample(img_x, flow_map)
        texture_encode, feat_encode = self.encoding_head(uv_images)
        feat_encode = feat_encode.view(feat_encode.size(0), -1)
        
        return uv_images, feat_encode
    
    def render_3d(self, pose_embeddings, center, scale, img_size, color, with_texture=False, flow=None, image=None, rot=False, location=None):
        self.device = "cuda"
        
        with torch.no_grad():
            pred_smpl_params, pred_cam = self.smpl_head(pose_embeddings)

        # Project the points to the image.
        # They are expected to be in [-1,1] range
        batch_size         = pose_embeddings.shape[0]
        dtype              = pred_cam.dtype
        focal_length       = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=self.device, dtype=dtype)
        
        smpl_output        = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_vertices      = smpl_output.vertices
        
        if(location is not None):
            pred_cam_t         = location[:, 0]
        else:
            pred_cam_t         = torch.stack([pred_cam[:,1], pred_cam[:,2], 2*focal_length[:, 0]/(pred_cam[:,0]*torch.tensor(scale, dtype=dtype, device=self.device) + 1e-9)], dim=1)
            pred_cam_t[:, :2] += torch.tensor(center-img_size/2., dtype=dtype, device=self.device) * pred_cam_t[:, [2]] / focal_length

        # initialize camera params and mesh faces for NMR
        K = torch.eye(3, device='cuda')
        K[0, 0] = K[1, 1]  = self.cfg.EXTRA.FOCAL_LENGTH
        K[2, 2] = 1
        K[1, 2] = K[0, 2]  = img_size/2.0
                                      
                                      
        K = K.unsqueeze(0)  # Our batch size is 1
        R = torch.eye(3, device='cuda').unsqueeze(0)
        t = torch.zeros(3, device='cuda').unsqueeze(0) 
        face_tensor = torch.tensor(self.faces_cpu.astype(np.int64), dtype=torch.long, device='cuda').unsqueeze_(0)
        face_tensor = face_tensor.repeat(batch_size, 1, 1)

        # transform vertices to world coordinates
        pred_cam_t_bs         = pred_cam_t.unsqueeze_(1).repeat(1, pred_vertices.size(1), 1)
        verts                 = pred_vertices  #+ pred_cam_t_bs
        if(rot):
            # bs x 6890 x 3,   3 x 3
            rot_y = torch.eye(3, device='cuda')*0
            rot_y[0,2] = 1
            rot_y[1,1] = 1
            rot_y[2,0] = -1
            pred_vertices = torch.matmul(pred_vertices, rot_y)
            verts         = pred_vertices + pred_cam_t_bs
                
        
        if(with_texture):
            # sample flow for all vertices
            flow_vert = torch.nn.functional.grid_sample(flow, self.uv_sampler.repeat(batch_size,1,1,1).cuda())
            flow_vert = flow_vert.view(flow_vert.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1).contiguous()

            # sample texture from the flow of all vertices
            texture_from_flow = sample_textures(flow_vert, image)            
            texture           = texture_from_flow.unsqueeze(4).expand(-1, -1, -1, -1, 6, -1) # B,F,T,T,T,3
        
        if(image is not None):
            rgb_from_pred = self.neural_renderer.visualize_all(verts.cpu().numpy(), pred_cam_t.cpu().numpy(), color, image)
        else:
            rgb_from_pred = 0

        
        
        ############ finding the 2d keypoints
        J_regressor = torch.from_numpy(np.load(self.cfg.SMPL.JOINT_REGRESSOR_H36M)).float().cuda()
        pred_keypoints_3d_smpl = torch.matmul(J_regressor, pred_vertices) #+ pred_t[i_]
        
        pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, [10, 9, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3], :]
        camera_center          = torch.zeros(batch_size, 2)
        pred_keypoints_2d_smpl = perspective_projection(pred_keypoints_3d_smpl,
                                                        rotation=torch.eye(3,).unsqueeze(0).expand(batch_size, -1, -1).cuda(),
                                                        translation=pred_cam_t.cuda(),
                                                        focal_length=focal_length / img_size,
                                                        camera_center=camera_center.cuda())  
        
        pred_keypoints_2d_smpl = (pred_keypoints_2d_smpl+0.5)*img_size
        
        return rgb_from_pred, pred_keypoints_2d_smpl, pred_keypoints_3d_smpl, pred_cam_t_bs
    
    
    def reset_nmr_sigle(self, image_size):
        self.neural_renderer = Renderer(focal_length=self.cfg.EXTRA.FOCAL_LENGTH, img_res=image_size, faces=self.faces_cpu)
    
    def reset_nmr(self, image_size):
        self.neural_renderer = Renderer(focal_length=self.cfg.EXTRA.FOCAL_LENGTH, img_res=image_size, faces=self.faces_cpu)

        
    def get_smpl_pose(self, pose_embeddings):
        self.device = "cuda"
        with torch.no_grad():
            pred_smpl_params, pred_cam = self.smpl_head(pose_embeddings)
            
        a = pred_smpl_params['global_orient']        
        b = pred_smpl_params['body_pose']        
        c = pred_smpl_params['betas']
        
        a = a.view(pose_embeddings.size(0), -1)
        b = b.view(pose_embeddings.size(0), -1)
        c = c.view(pose_embeddings.size(0), -1)
        
        return torch.cat([a, b, c], -1)
       
