from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import transforms, utils

import os
import cv2
import wandb
import random
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Code adjusted from: https://github.com/mkocabas/VIBE/blob/master/lib/data_utils/img_utils.py

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, trans_inv

def generate_image_patch(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    img = cvimg.copy()
    # c = center.copy()
    img_height, img_width, img_channels = img.shape

    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans, trans_inv = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    return img_patch, trans, trans_inv




class PoseTrack(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, window=1, frame_length=10, img_size=256, max_ids=5, given_id=-1, key_frame=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.mean             = np.array([123.675, 116.280, 103.530])
        self.std              = np.array([58.395, 57.120, 57.375])
        
        self.key_frame        = key_frame
        self.window           = window
        self.frame_length     = frame_length
        self.img_size         = img_size
        self.max_ids          = max_ids
        self.data             = []
        self.dataset          = []
        self.root_dir_2018    = "_DATA/Posetrack_2018/"
        
        self.tracking_data    = np.load("_DATA/posetrack_train.npz", allow_pickle=True)


        for i_, track_ in enumerate(self.tracking_data['arr_0']):
            for i in range((len(track_)-self.frame_length)//self.window + 1):
                self.data.append(track_[i*self.window:(i*self.window)+frame_length])
                

                

    def process_image(self, img, center, scale):
        img, _, _ = generate_image_patch(img, center[0], center[1], scale, scale, self.img_size, self.img_size, False, 1.0, 0.0)
        img = img[:, :, ::-1].copy().astype(np.float32)
        img_n = img[:, :, ::-1].copy().astype(np.float32)
        for n_c in range(3):
            img_n[:, :, n_c] = (img_n[:, :, n_c] - self.mean[n_c]) / self.std[n_c]
        return torch.from_numpy(np.transpose(img_n, (2, 0, 1)))
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_names        = []
        ids              = torch.zeros(self.frame_length, self.max_ids) + -1
        centers          = torch.zeros(self.frame_length, self.max_ids, 2)
        scales           = torch.zeros(self.frame_length, self.max_ids, 2) 
        bboxs            = torch.zeros(self.frame_length, self.max_ids, 4) 
        keypoint_2d      = torch.zeros(self.frame_length, self.max_ids, 15, 2) 
        keypoint_3d      = torch.zeros(self.frame_length, self.max_ids, 15, 3) 
        keypoint_3t      = torch.zeros(self.frame_length, self.max_ids, 15, 3) 
            
        pose_emb         = torch.zeros(self.frame_length, self.max_ids, 2048) 
        apper_emb        = torch.zeros(self.frame_length, self.max_ids, 512)                
        smpl_emb         = torch.zeros(self.frame_length, self.max_ids, 226)                
                
        frames_          = self.data[idx]
        for t, f in enumerate(frames_):
            root_dir     = self.root_dir_2018
            img_name     = os.path.join(root_dir , f[0])

            img_names.append(img_name)
            ids_         = f[1]
            center_      = f[3]
            scale_       = f[4]
            bbox_        = f[5]
                
                
            pose_        = f[6]
            apper_       = f[7]
            smpl_        = f[10]
            
            for i, idx__ in enumerate(ids_[:self.max_ids]):
                if(scale_[i][0]==0 or scale_[i][1]==0): 
                    ids[t, i] = -1 
                    continue

                ids[t, i] = idx__
                if(scale_[i][0]<50 or scale_[i][1]<100):
                    ids[t, i] = -1
                
                image_width             = f[9][i][1] 
                centers[t, i, :]        = torch.from_numpy(center_[i] )
                scales[t, i, :]         = torch.from_numpy(scale_[i] ) 
                bboxs[t, i, :]          = torch.from_numpy(bbox_[i] )
                keypoint_2d[t, i, :, :] = torch.from_numpy(f[8][0][i])
                keypoint_3d[t, i, :, :] = torch.from_numpy(f[8][1][i]) 
                keypoint_3t[t, i, :, :] = torch.from_numpy(f[8][2][i]) 
                keypoint_3d[t, i, :, :] = keypoint_3t[t, i, :, :] + keypoint_3d[t, i, :, :]
                
                pose_emb[t,  i, :]      = torch.from_numpy(pose_[i])
                apper_emb[t,  i, :]     = torch.from_numpy(apper_[i])
                smpl_emb[t,  i, :]      = torch.from_numpy(smpl_[i])

                
        return 0, ids, centers, scales, bboxs, img_names, pose_emb, apper_emb, keypoint_3d, smpl_emb
