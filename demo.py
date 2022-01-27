# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import torchvision.models as models

import os
import json
import copy
import shutil
import heapq
import argparse
import pickle
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from yacs.config import CfgNode as CN
from tqdm import tqdm

from models.hmar import HMAR
from utils.utils_dataset import process_image
from test_t3dp import test_tracker
from HMAR_tracker import HMAR_tracker

import time
import datetime
import math
from pytube import YouTube

class FrameExtractor():
    '''
    Class used for extracting frames from a video file.
    '''
    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap    = cv2.VideoCapture(video_path)
        self.n_frames   = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps        = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
        
    def get_video_duration(self):
        duration = self.n_frames/self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')
        
    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')
        
    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext = '.jpg', max_frames=1000):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path); print(f'Created the following directory: {dest_path}')
        
        frame_cnt = 0; img_cnt = 0
        while self.vid_cap.isOpened():
            success,image = self.vid_cap.read() 
            if not success: break
            if frame_cnt % every_x_frame == 0:
                img_path = os.path.join(dest_path, ''.join([img_name,  '%06d' % (img_cnt+1), img_ext]))
                cv2.imwrite(img_path, image)  
                img_cnt += 1
            frame_cnt += 1
            if(max_frames<img_cnt): break
        self.vid_cap.release()
        cv2.destroyAllWindows()






def process_image_simple(img):
    mean = np.array([123.675, 116.280, 103.530])
    std = np.array([58.395, 57.120, 57.375])
    
    img = img[:, :, ::-1].copy().astype(np.float32)
    img_n = img[:, :, ::-1].copy().astype(np.float32)
    for n_c in range(3):
        img_n[:, :, n_c] = (img_n[:, :, n_c] - mean[n_c]) / std[n_c]
    return torch.from_numpy(np.transpose(img_n, (2, 0, 1)))



def run_detection(image_path):
    
    time_ = []
    cfg   = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    files          = glob.glob(os.path.join(image_path, '*.jpg'))
    files.sort()

    for f in tqdm(files):
        im = cv2.imread(f)
        t1 = time.time()
        outputs = predictor(im)
        time_.append(time.time()-t1)
        instances = outputs['instances']
        out_npz = os.path.join(image_path + "/detections", '%s.npz' % f.split('/')[-1][:-4])
        np.savez(out_npz, classes=instances.pred_classes.cpu().numpy(), scores=instances.scores.cpu().numpy(), boxes=instances.pred_boxes.tensor.cpu().numpy(), masks=instances.pred_masks.cpu().numpy())

    # extract masks
    npzs = glob.glob(os.path.join(image_path + "/detections", '*.npz'))
    npzs.sort()
    max_count = 0
    for npz_i in npzs:
        npz   = np.load(npz_i)
        count = 0
        for i in range(npz['classes'].shape[0]):
            if npz['classes'][i] > 0:
                continue
            cv2.imwrite(os.path.join(image_path + "/detections", '%s_%02d.png' % (npz_i.split('/')[-1][:-4], count)), npz['masks'][i].astype(int)*255)
            count = count + 1
        max_count = max(count, max_count)

    # store full video npz
    imgnames_, masknames_, centers_, scales_, instances_, confs_ = [], [], [], [], [], []
    instances        = -np.ones([1, len(npzs), max_count]).astype(int)
    groups           = -np.ones([1, len(npzs)]).astype(int)
    counter_instance = 0

    for fi, npz in enumerate(npzs):
        npz_contents = np.load(npz)
        ui = 0
        for bbox, class_id, mask, score in zip(npz_contents['boxes'], npz_contents['classes'], npz_contents['masks'], npz_contents['scores']):
            if class_id > 0:
                continue

            imgname  = image_path + '%06d.jpg' % (fi+1)
            maskname = image_path + "/detections/" + '%06d_%02d.png' % ((fi+1), ui)

            # scale and center
            center = [(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2]
            scale = [1.2*(bbox[2] - bbox[0]), 1.2*(bbox[3] - bbox[1])]

            # instance
            instances[0, fi, ui] = counter_instance

            # increase counter
            counter_instance = counter_instance + 1
            ui = ui + 1

            imgnames_.append(imgname)
            masknames_.append(maskname)
            centers_.append(center)
            scales_.append(scale)
            confs_.append(score)
    
    video_npz = os.path.join(image_path, 'detections.npz')
    np.savez(video_npz, 
             imgname=imgnames_,
             maskname=masknames_,
             center=centers_,
             scale=scales_,
             conf=confs_,
             instances=instances)
    
    return time_


def run_hmar(video_path):
    time_           = []
    config          = os.path.join('utils/config.yaml')
    checkpoint      = '_DATA/t3dp_hmar.pt'
        
    RGB_tuples      = np.random.uniform(0, 255, size=(40, 3)) 
     
    HMAR_model      = HMAR(config)
    checkpoint      = torch.load(checkpoint)
    state_dict_filt = {k: v for k, v in checkpoint['model'].items() if not("perceptual_loss" in k)}
    state_dict_filt = { k.replace('smplx', 'smpl'): v for k, v in state_dict_filt.items() }
    
    HMAR_model.load_state_dict(state_dict_filt, strict=True)
    HMAR_model.cuda()
    HMAR_model.eval()
    
    res             = 256
    HMAR_model.reset_nmr(res)

    video_data      = np.load(video_path + "/detections.npz"); is_gt = False # val data
    video_seq       = video_data['instances']
    base_path       = video_path 
    mask_path       = video_path + '/detections/'
    dataset         = 'demo'
    
    video_seq_      = []
    for video_id, video in enumerate(video_seq):
        
        track                             = {}
        old_image_size                    = 10
        
        frame_num = 0
        for frame in tqdm(video):
            try:    shot = video_data['shots'][video_id][frame_num]; frame_num += 1
            except: shot = 0
                
            f_loc                         = frame!=-1    
            frame_ids                     = frame[f_loc]
            if(len(frame_ids)==0): continue
                

            frame_ids_   = []
            frame_ids_gt = []
            for kl in frame_ids:
                frame_ids_.append(kl)
                gt_loc_ = np.where(frame==kl)[0]
                frame_ids_gt.append(gt_loc_[0])
                

            frame_path                    = video_data['imgname'][int(frame_ids[0])] 
            frame_name                    = frame_path.split("/")[-1]
            video_name                    = frame_path.split("/")[-2] 
            
            image                         = cv2.imread(base_path + "/" + frame_name)
            img_height, img_width, _      = image.shape
            new_image_size                = max(img_height, img_width)
            delta_w                       = new_image_size - img_width
            delta_h                       = new_image_size - img_height
            top, bottom                   = delta_h//2, delta_h-(delta_h//2)
            left, right                   = delta_w//2, delta_w-(delta_w//2)

            if(video_name in track.keys()): track[video_name][frame_name] = {}
            else:                           track[video_name] = {}; track[video_name][frame_name] = {}; video_seq_.append(video_name)
                
            for idx, det_person in enumerate(frame_ids):   
                id_                       = int(det_person)
                id_gt                     = int(frame_ids_gt[idx])
                center_                   = video_data['center'][id_]
                scale_                    = video_data['scale'][id_]
                conf_                     = video_data['conf'][id_] if not(is_gt) else 1
                x1                        = center_[0] - scale_[0]/2.0
                y1                        = center_[1] - scale_[1]/2.0
                x2                        = center_[0] + scale_[0]/2.0
                y2                        = center_[1] + scale_[1]/2.0
                w                         = x2-x1; h = y2-y1
                center_                   = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
                scale_                    = np.array([w, h])
                image_tmp                 = process_image(image, center_, 1.0*np.max(scale_))

                with torch.no_grad():
                    ratio                 = 1.0/int(new_image_size)*res
                    t1                    = time.time()
                    pose_embedding, appe_embedding, flow, uv_map                               = HMAR_model(image_tmp.unsqueeze(0).cuda())
                    time_.append(time.time()-t1)
                    rendered_image, pred_keypoints_2d, pred_keypoints_3d, translation          = HMAR_model.render_3d(pose_embedding, (center_ + [left, top])*ratio, max(scale_)*ratio, res, np.array([[150,0,0]])/255.)
                    smpl_parameters                                                            = HMAR_model.get_smpl_pose(pose_embedding)

                 
                track[video_name][frame_name][idx+1]                   = {}
                track[video_name][frame_name][idx+1]['score']          = conf_
                track[video_name][frame_name][idx+1]['bbox']           = np.array([x1, y1, w, h])
                track[video_name][frame_name][idx+1]['center']         = center_
                track[video_name][frame_name][idx+1]['scale']          = scale_
                track[video_name][frame_name][idx+1]['keypoints_3d']   = pred_keypoints_3d.cpu().numpy()
                track[video_name][frame_name][idx+1]['keypoints_3t']   = translation[0, :15, :].cpu().numpy()
                track[video_name][frame_name][idx+1]['keypoints_2d']   = pred_keypoints_2d.cpu().numpy()
                track[video_name][frame_name][idx+1]['image_size']     = np.array([img_height, img_width])
                track[video_name][frame_name][idx+1]['appe_embedding'] = appe_embedding.detach().cpu().numpy()
                track[video_name][frame_name][idx+1]['pose_embedding'] = pose_embedding.detach().cpu().numpy()

        list_of_frames_ = os.listdir(base_path)
        list_of_frames  = [f for f in list_of_frames_ if ".jpg" in f]
        
        for frame_ in list_of_frames:
            if(frame_ not in track[video_name].keys()):
                track[video_name][frame_]             = {}
                track[video_name][frame_][1]          = {}
                track[video_name][frame_][1]['score'] = -1
                
        with open(video_path + '/hmar_' + video_name + '.pickle', 'wb') as handle:
            pickle.dump(track, handle, protocol=pickle.HIGHEST_PROTOCOL)
                 
    return time_  
            
            
if __name__ == '__main__':

    
    
    
    YOUTUBE_ID     = 'xEH_5T9jMVU'  
    video_folder   = "video_"+ YOUTUBE_ID +"/"
    dataset_path   = "_DATA/DEMO/" 
    
#     os.system("rm -rf "+"_DATA/DEMO/"+video_folder)
#     os.system("mkdir _DATA/DEMO/")
#     os.system("mkdir _DATA/DEMO/"+video_folder)
#     os.system("mkdir _DATA/DEMO/"+video_folder+"/detections")

#     video = YouTube('https://www.youtube.com/watch?v=' + YOUTUBE_ID)
#     print('Summary:')
#     print(f'Title: {video.title}')
#     print(f'Duration: {video.length / 60:.2f} minutes')
#     # print(f'Rating: {video.rating:.2f}')
#     print(f'# of views: {video.views}')
#     print(video.streams.all())
#     video.streams.get_by_itag(18).download(output_path = dataset_path + video_folder, filename="youtube.mp4")
#     fe = FrameExtractor(dataset_path + video_folder + "/youtube.mp4")
#     print(fe.n_frames)
#     print(fe.get_video_duration())
#     fe.extract_frames(every_x_frame=1, img_name='', dest_path=dataset_path + video_folder + "/", max_frames=100000)
    
    
#     detection_time_    = run_detection(dataset_path + video_folder)
#     hmar_time_         = run_hmar(dataset_path + video_folder)

    
    parser = argparse.ArgumentParser(description='T3PO Tracker')
    parser.add_argument('--dataset', type=str, default='val')

    opt                = parser.parse_args()
    opt.storage_folder = "Videos_Final_DEMO"    
    opt.dataset        = "demo"
    opt.dataset_path   = "_DATA/DEMO/"
    opt.th_x           = 20000
    opt.past_x         = 100
    opt.max_age_x      = 100
    opt.n_init_x       = 5
    opt.max_ids_x      = 10
    opt.window_x       = 10
    opt.metric_x       = "euclidean_min"
    opt.render         = False
    opt.save           = False
    opt.downsample     = 1
    opt.videos_seq     = ["video_"+ YOUTUBE_ID]
    
    
    hmar_tracker       = HMAR_tracker(mode="APK", betas=[1.0,1.0,1.0])
    path_model         = os.path.join('_DATA/t3dp_transformer.pth')      
   
    prev_best          = torch.load(path_model)
    print("loading from ", prev_best['epoch'])
    hmar_tracker.load_state_dict(prev_best['model'], strict=True)    
    
    hmar_tracker.cuda()
    hmar_tracker.eval()
    
    t3dp_time_         = test_tracker(opt, hmar_tracker)

    # print("Mask RCNN runtime : ", np.sum(detection_time_))    
    # print("HMAR runtime : ", np.sum(hmar_time_))    
    print("T3DP runtime : ", np.sum(t3dp_time_))