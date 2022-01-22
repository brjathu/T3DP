import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageDraw, ImageFont, ImageColor

import os
import json
import joblib
import copy
import heapq
import argparse
import pickle
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from yacs.config import CfgNode as CN

RGB_tuples      = np.vstack([np.loadtxt("utils/colors.txt", skiprows=1) , np.random.uniform(0, 255, size=(1000, 3))])
b               = np.where(RGB_tuples==0)
RGB_tuples[b]   = 1


def refine_visuals(a2):
    track_data = {}
    for frame_ in a2.keys():
        frame_data = a2[frame_]
        for tid in range(len(frame_data[0])):
            tid_ = frame_data[0][tid]
            if(tid_ in track_data.keys()):
                track_data[tid_][frame_] = [frame_data[1][tid], frame_data[2], frame_data[3][tid], frame_data[4][tid], frame_data[5][tid], frame_data[6][tid], frame_data[7][tid], frame_data[8][tid], frame_data[9][tid]]
            else:
                track_data[tid_] = {}
                track_data[tid_][frame_] = [frame_data[1][tid], frame_data[2], frame_data[3][tid], frame_data[4][tid], frame_data[5][tid], frame_data[6][tid], frame_data[7][tid], frame_data[8][tid], frame_data[9][tid]]




    refined_track_data = {}
    for tid in track_data.keys():

        t_time  = []
        t_frame = []
        for frame_ in track_data[tid].keys():
            t_time.append(track_data[tid][frame_][1])        
            t_frame.append(frame_)

        if(len(t_time)<10): continue
        refined_track_data[tid] = {}
        for i, (t_, f_) in enumerate(zip(t_time[:-1], t_frame[:-1])):
            tx = t_time[i+1]
            fx = t_frame[i+1]
            if(tx!=t_+1):
                center_start = track_data[tid][f_][3]
                center_end   = track_data[tid][fx][3]

                scale_start  = track_data[tid][f_][4]
                scale_end    = track_data[tid][fx][4]
                
                loca_start   = track_data[tid][f_][8]
                loca_end     = track_data[tid][fx][8]
                i = 0
                for t_new in range(t_, tx):
                    # f_new = "img_" + str(t_new).zfill(6) + ".jpg"
                    f_new = str(t_new).zfill(6) + ".jpg"
                    center_interpolated = center_start + (center_end-center_start)/(tx-t_)*(i)
                    scale_interpolated  = scale_start +(scale_end-scale_start)/(tx-t_)*(i)
                    loca_interpolated   = loca_start +(loca_end-loca_start)/(tx-t_)*(i)
                    refined_track_data[tid][f_new] = [track_data[tid][f_][0], track_data[tid][f_][1], track_data[tid][f_][2], center_interpolated, scale_interpolated, track_data[tid][f_][5], track_data[tid][f_][6], track_data[tid][f_][7], loca_interpolated, 0]
                    i += 1
            else:
                refined_track_data[tid][f_] = [track_data[tid][f_][0], track_data[tid][f_][1], track_data[tid][f_][2], track_data[tid][f_][3], track_data[tid][f_][4], track_data[tid][f_][5], track_data[tid][f_][6], track_data[tid][f_][7], track_data[tid][f_][8], 1]



    refined_eval_dic = {}
    for tid in refined_track_data.keys():
        for frame_ in refined_track_data[tid].keys():

            if(frame_ in refined_eval_dic.keys()):
                refined_eval_dic[frame_][0].append(tid)
                refined_eval_dic[frame_][1].append(refined_track_data[tid][frame_][0])

            else:
                refined_eval_dic[frame_] = [[tid], [refined_track_data[tid][frame_][0]], refined_track_data[tid][frame_][1]]

    refined_visuals_dic = {}
    for tid in refined_track_data.keys():
        for frame_ in refined_track_data[tid].keys():
            refined_track_data[tid][frame_]

            if(frame_ in refined_visuals_dic.keys()):
                refined_visuals_dic[frame_][0].append(tid)
                refined_visuals_dic[frame_][1].append(refined_track_data[tid][frame_][0])
                refined_visuals_dic[frame_][3].append(refined_track_data[tid][frame_][2])
                refined_visuals_dic[frame_][4].append(refined_track_data[tid][frame_][3])
                refined_visuals_dic[frame_][5].append(refined_track_data[tid][frame_][4])
                refined_visuals_dic[frame_][6].append(refined_track_data[tid][frame_][5])
                refined_visuals_dic[frame_][7].append(refined_track_data[tid][frame_][6])
                refined_visuals_dic[frame_][8].append(refined_track_data[tid][frame_][7])
                refined_visuals_dic[frame_][9].append(refined_track_data[tid][frame_][8])
                refined_visuals_dic[frame_][10].append(refined_track_data[tid][frame_][9])

            else:
                refined_visuals_dic[frame_] = [[tid], 
                                               [refined_track_data[tid][frame_][0]], 
                                               refined_track_data[tid][frame_][1], 
                                               [refined_track_data[tid][frame_][2]], 
                                               [refined_track_data[tid][frame_][3]], 
                                               [refined_track_data[tid][frame_][4]], 
                                               [refined_track_data[tid][frame_][5]], 
                                               [refined_track_data[tid][frame_][6]], 
                                               [refined_track_data[tid][frame_][7]],
                                               [refined_track_data[tid][frame_][8]],
                                               [refined_track_data[tid][frame_][9]]
                                              ]
                
    return refined_visuals_dic, refined_eval_dic




def make_video(HMAR_model, save, render, opt, video_name, final_visuals_dic):
    
    t_ = 0
    old_image_size = 10
    start = 0
    for frame_ in final_visuals_dic.keys():
        
        cv_image = cv2.imread(opt.dataset_path + "/" + video_name + "/" + frame_)
            
        img_height, img_width, _  = cv_image.shape
        frame_data = final_visuals_dic[frame_]
        visual_ids = np.array(frame_data[8])
        

        try:     pose_data   = torch.stack(frame_data[3])
        except:  pose_data   = frame_data[3]
        
        try:     center_data = np.array(frame_data[4])
        except : center_data = frame_data[4]
            
        try:     color_data  = np.array(frame_data[6])
        except : color_data  = frame_data[6]
            
        try:     scale_data  = np.array(frame_data[5])
        except : scale_data  = frame_data[5]
            
            
        # try:     loca_data   = torch.stack(frame_data[9])
        # except : loca_data   = frame_data[9]
            
        new_image_size            = max(img_height, img_width)
        if(new_image_size!=old_image_size): HMAR_model.reset_nmr(new_image_size); old_image_size = new_image_size
        delta_w                   = new_image_size - img_width
        delta_h                   = new_image_size - img_height
        top, bottom               = delta_h//2, delta_h-(delta_h//2)
        left, right               = delta_w//2, delta_w-(delta_w//2)
        resized_image             = cv2.copyMakeBorder(cv_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        resized_image_bbox        = copy.deepcopy(resized_image)

        if(start==0): 
            start = 1
            fourcc         = cv2.VideoWriter_fourcc(*'mp4v')
            video_file     = cv2.VideoWriter("out/" + opt.storage_folder + "/" + video_name + ".mp4", fourcc, 15, frameSize=(2*img_width//opt.downsample, img_height//opt.downsample))
            # fourcc     = cv2.VideoWriter_fourcc(*'MJPG')
            # video_file = cv2.VideoWriter()

        ### visiualize the tracks using 3D rendering or with bboxes
        if(render):
            with torch.no_grad():
                if(visual_ids.shape[0]==0): 
                    rendered_image = resized_image_bbox.astype(float)
                    rendered_image = rendered_image/255.0
                else:
                        
                    loc_x = np.array(frame_data[10])>0.5
                    if(np.sum(loc_x)>0):
                        rendered_image, _, _, _   = HMAR_model.render_3d(pose_data[loc_x].cuda(), center_data[loc_x] + [left, top], np.max(scale_data[loc_x], axis=1), max(img_height, img_width), color_data[loc_x], image=resized_image_bbox/255.0)
                    else:
                        rendered_image = resized_image_bbox.astype(float)
                        rendered_image = rendered_image/255.0
                    
        else:
            for track_id in visual_ids:
                bbox_ = frame_data[7][track_id[0]]
                cv2.rectangle(resized_image_bbox, (int(bbox_[0])+left, int(bbox_[1])+top), (int(bbox_[2])+left, int(bbox_[3])+top), [int(i) for i in RGB_tuples[track_id[1]]], 4)
                cv2.putText(resized_image_bbox, str(track_id), (int(bbox_[2]-60),int(bbox_[1]+60)), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,255))


        resized_image = torch.from_numpy(np.array(resized_image)/255.).cuda()
        resized_image = resized_image.unsqueeze(0)
        resized_image = resized_image.permute(0, 3, 1, 2)
        resized_image = resized_image[:, [2,1,0], :, :] 

        if(render):
            rendered_image = torch.from_numpy(np.array(rendered_image)).cuda()
            rendered_image = rendered_image.unsqueeze(0)
            rendered_image = rendered_image.permute(0, 3, 1, 2)
            rendered_image = rendered_image[:, [2,1,0], :, :] 

            grid_img = make_grid(torch.cat([resized_image[:, :, top:top+img_height, left:left+img_width], rendered_image[:, :, top:top+img_height, left:left+img_width], ], 0), nrow=10)

        else:

            resized_image_bbox = torch.from_numpy(np.array(resized_image_bbox)/255.).cuda()
            resized_image_bbox = resized_image_bbox.unsqueeze(0)
            resized_image_bbox = resized_image_bbox.permute(0, 3, 1, 2)
            resized_image_bbox = resized_image_bbox[:, [2,1,0], :, :] 

            grid_img = make_grid(torch.cat([resized_image[:, :, top:top+img_height, left:left+img_width], resized_image_bbox[:, :, top:top+img_height, left:left+img_width], ], 0), nrow=10)


        grid_img = grid_img[[2,1,0], :, :] 
        ndarr    = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        cv_ndarr = cv2.resize( ndarr , (2*img_width//opt.downsample, img_height//opt.downsample) )
        video_file.write(cv_ndarr)
        
        t_ += 1
    
    try: video_file.release()
    except: print("no videos")