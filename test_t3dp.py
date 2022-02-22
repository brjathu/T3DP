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
import time 
import numpy as np
from PIL import Image
from tqdm import tqdm
from yacs.config import CfgNode as CN

from models.hmar import HMAR
from utils.utils_measure import AverageMeter
from utils.make_video import refine_visuals, make_video

from HMAR_tracker import HMAR_tracker

from deep_sort_ import nn_matching
from deep_sort_.detection import Detection
from deep_sort_.tracker import Tracker


    
RGB_tuples      = np.vstack([np.loadtxt("utils/colors.txt", skiprows=1) , np.random.uniform(0, 255, size=(1000, 3))])
b               = np.where(RGB_tuples==0)
RGB_tuples[b]   = 1


        

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):   return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.') 
        
        
def test_tracker(opt, hmar_tracker):

    config          = os.path.join('utils/config.yaml')
    checkpoint      = '_DATA/t3dp_hmar.pt'

    HMAR_model      = HMAR(config)
    checkpoint      = torch.load(checkpoint)
    state_dict_filt = {k: v for k, v in checkpoint['model'].items() if not("perceptual_loss" in k)} 
    state_dict_filt = {k.replace('smplx', 'smpl'): v for k, v in state_dict_filt.items()}
    HMAR_model.load_state_dict(state_dict_filt, strict=True)
    HMAR_model.cuda()
    HMAR_model.eval()
        

    try:    os.system("mkdir out");os.system("mkdir out/" + opt.storage_folder); os.system("mkdir out/" + opt.storage_folder + "/results")
    except: pass

    for num_video, video_name in tqdm(enumerate(opt.videos_seq)):
       
        if(opt.dataset=="demo"): track = joblib.load(opt.dataset_path + '/' + str(video_name) + '/hmar_' + video_name + '.pickle')
        else:                    track = joblib.load('_DATA/detections/hmar_' + opt.dataset + "_" + str(video_name) + '.pickle')

        final_results        = []
        final_results_dic    = {}
        final_visuals_dic    = {}
        
        sequence = track.keys()
        for video in tqdm(sequence):
            metric           = nn_matching.NearestNeighborDistanceMetric(opt.metric_x, opt.th_x, opt.past_x)
            tracker          = Tracker(metric, max_iou_distance=0.9, max_age=opt.max_age_x, n_init=opt.n_init_x)  


            frame_list       = sorted(list(track[video].keys()))
            frame_length     = len(frame_list)
            max_ids          = opt.max_ids_x
            img_size         = 256
            person_id        = torch.zeros(frame_length, max_ids) + -1
            center           = torch.zeros(frame_length, max_ids, 2)
            scale            = torch.zeros(frame_length, max_ids, 2) 
            bbox             = torch.zeros(frame_length, max_ids, 4) 
            keypoints_2d     = torch.zeros(frame_length, max_ids, 15, 2) 
            keypoints_3d     = torch.zeros(frame_length, max_ids, 15, 3) 
            keypoints_3t     = torch.zeros(frame_length, max_ids, 15, 3) 
            pose_emb         = torch.zeros(frame_length, max_ids, 2048) 
            appe_emb         = torch.zeros(frame_length, max_ids, 512) 


            for frame_idx, frame in enumerate(frame_list):

                for idx in range(min(max_ids, len(track[video][frame]))):  
                    person_data                     = track[video][frame][idx+1]
                    if(person_data['score']>0.5):
                        person_id[frame_idx, idx]           = 0
                        center[frame_idx, idx, :]           = torch.from_numpy(np.array(person_data['center']))
                        scale[frame_idx, idx, :]            = torch.from_numpy(np.array(person_data['scale']))
                        bbox[frame_idx, idx, :]             = torch.from_numpy(np.array([person_data['bbox'][0], person_data['bbox'][1], person_data['bbox'][0]+person_data['bbox'][2], person_data['bbox'][1]+person_data['bbox'][3]]))
                        keypoints_2d[frame_idx, idx, :, :]  = torch.from_numpy(np.array(person_data['keypoints_2d']))
                        keypoints_3t[frame_idx, idx, :, :]  = torch.from_numpy(np.array(person_data['keypoints_3t']))
                        keypoints_3d[frame_idx, idx, :, :]  = torch.from_numpy(np.array(person_data['keypoints_3t'])) + torch.from_numpy(np.array(person_data['keypoints_3d']))
                        pose_emb[frame_idx, idx, :]         = torch.from_numpy(person_data['pose_embedding'])
                        appe_emb[frame_idx, idx, :]         = torch.from_numpy(person_data['appe_embedding'])


                

                        
            BS, T, P   = 1, frame_length, max_ids
            window     = frame_length//opt.window_x
            start_     = 0; start_2    = 0
            
            for w_ in range(frame_length//window):
                with torch.no_grad():
                    for i in range(100):
                        output, _  = hmar_tracker.forward(BS, window, P,  [pose_emb[w_*window:(w_+1)*window].unsqueeze(0).cuda(), appe_emb[w_*window:(w_+1)*window].unsqueeze(0).cuda()], 
                                                                          person_id[w_*window:(w_+1)*window].unsqueeze(0), 
                                                                          bbox[w_*window:(w_+1)*window].unsqueeze(0), 
                                                                          keypoints_3d[w_*window:(w_+1)*window].unsqueeze(0))   
                        embeddings = output["output_embeddings"] 
                embeddings = embeddings.view(BS, window, P, -1)
                for t in list(range(window)):
                    t_   = t + w_*window
                    loc_ = np.where(person_id[w_*window:(w_+1)*window][t]!=-1)[0]
                    embeddings_normalized  = embeddings[0, t, loc_].cpu().numpy()

                    detections       = []
                    detection_filter = []
                    for m in range(len(bbox[w_*window:(w_+1)*window][t][loc_])):
                        w = bbox[w_*window:(w_+1)*window][t][loc_][m][2] - bbox[w_*window:(w_+1)*window][t][loc_][m][0]
                        h = bbox[w_*window:(w_+1)*window][t][loc_][m][3] - bbox[w_*window:(w_+1)*window][t][loc_][m][1]
                        if(h>120 and w>60):
                            det = Detection([bbox[w_*window:(w_+1)*window][t][loc_][m][0], bbox[w_*window:(w_+1)*window][t][loc_][m][1], w, h], 1.0, embeddings_normalized[m])
                            detections.append(det); detection_filter.append(m)
                    
                    tracked_ids_ = []; tracked_bbox_ = []
                    
                    tracker.predict()
                    if(len(detections)>0): 
                        matches = tracker.update(detections)
                    visual_ids   = []
                    for tracks_ in tracker.tracks:
                        if(tracks_.time_since_update!=0):  continue
                        track_id     = tracks_.track_id
                        detection_id = tracks_.detection_id[-1]
                        bbox_        = tracks_.bbox[-1]
                        visual_ids.append([detection_id, track_id])    
                        tracked_ids_.append(track_id)
                        tracked_bbox_.append(bbox_)
                            
                    visual_ids = np.array(visual_ids)  
                    final_results_dic.setdefault(frame_list[t_], [tracked_ids_, tracked_bbox_, t_]) 
                    if(visual_ids.shape[0]!=0): 
                        final_visuals_dic.setdefault(frame_list[t_], [tracked_ids_, tracked_bbox_, t_, 
                                                                  pose_emb[w_*window:(w_+1)*window][t][loc_][detection_filter][visual_ids[:, 0]], 
                                                                  np.array(center[w_*window:(w_+1)*window][t][loc_][detection_filter][visual_ids[:, 0]]), 
                                                                  np.array(scale[w_*window:(w_+1)*window][t][loc_][detection_filter][visual_ids[:, 0]]), 
                                                                  np.array(RGB_tuples[visual_ids[:, 1]])/255.0,
                                                                  bbox[w_*window:(w_+1)*window][t][loc_][detection_filter],
                                                                  visual_ids,
                                                                  keypoints_3t[w_*window:(w_+1)*window][t][loc_][detection_filter][visual_ids[:, 0]],
                                                                 ]) 
                    else:
                        final_visuals_dic.setdefault(frame_list[t_], [tracked_ids_, tracked_bbox_, t_, [], [], [], [], [], visual_ids, []]) 
                        
            save_loc = video_name.split("/")[0] + "______" + video_name.split("/")[1] if("AVA" in opt.dataset_path.split("/")[-2]) else video_name
            joblib.dump(final_results_dic, "out/" + opt.storage_folder + "/results/" + save_loc + ".pkl")
            
            if(opt.save): 
                new_visuals_dic, refined_eval_dic = refine_visuals(final_visuals_dic)
                make_video(HMAR_model, opt.save, opt.render, opt, video_name, new_visuals_dic)                 
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='T3DP Tracker')
    parser.add_argument('--dataset', type=str, default='posetrack')
    parser.add_argument('--dataset_path', type=str, default="/_DATA/Posetrack_2018/")
    parser.add_argument('--storage_folder', type=str, default="Videos_Final")
    parser.add_argument('--th_x', type=int, default=20000)
    parser.add_argument('--past_x', type=int, default=20)
    parser.add_argument('--max_age_x', type=int, default=20)
    parser.add_argument('--n_init_x', type=int, default=5)
    parser.add_argument('--max_ids_x', type=int, default=50)
    parser.add_argument('--window_x', type=int, default=1)
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--metric_x', type=str, default="euclidean_min")
    parser.add_argument('--render', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--save', type=str2bool, nargs='?', const=True, default=True)
    opt = parser.parse_args()

    if(opt.dataset=="posetrack"): opt.videos_seq = np.load("_DATA/posetrack.npy")
    if(opt.dataset=="mupots"):    opt.videos_seq = np.load("_DATA/mupots.npy")
  

    hmar_tracker    = HMAR_tracker(mode="APK", betas=[1.0,1.0,1.0])
    path_model      = os.path.join('_DATA/t3dp_transformer.pth')   #  APK, HMAR, posetrack

    prev_best       = torch.load(path_model)
    print("loading from ", prev_best['epoch'])
    hmar_tracker.load_state_dict(prev_best['model'], strict=True)    
    
    hmar_tracker.cuda()
    hmar_tracker.eval()
    test_tracker(opt, hmar_tracker)






