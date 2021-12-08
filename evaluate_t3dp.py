import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
# from matplotlib.pyplot import imshow
import cv2
import torch
import json
import pickle
import copy
from tqdm import tqdm


import motmetrics as mm
from utils.losses import ReIDLoss
# from pycocotools.coco import COCO

import sys
from tqdm import tqdm
import joblib
from torchvision.utils import save_image, make_grid


RGB_tuples      = np.vstack([np.loadtxt("utils/colors.txt", skiprows=1) , np.random.uniform(0, 255, size=(1000, 3))])
b               = np.where(RGB_tuples==0)
RGB_tuples[b]   = 1

    
# # # #########################################################################################################
# # # #########################################################################################################
# # # ###############################     attach predicted data - T3PO        #################################
# # # #########################################################################################################
# # # #########################################################################################################


    
    
def evaluate_trackers(results_dir, method="phalp", dataset="posetrack", make_video=0):   
    if(dataset=="posetrack"): data_gt = joblib.load('_DATA/posetrack_gt.pickle')     ; base_dir = "_DATA/Posetrack_2018/"
    if(dataset=="mupots"):    data_gt = joblib.load('_DATA/mupots_gt.pickle')        ; base_dir = "_DATA/MuPoTs/"
    if(dataset=="ava"):       data_gt = joblib.load('_DATA/ava_gt.pickle')           ; base_dir = "_DATA/AVA/"
        
    data_all              = {}
    total_annoated_frames = 0
    total_detected_frames = 0
    
    if(method=='t3dp'):
        for video_ in data_gt.keys():
            
            try:                
                if(dataset=="ava"): T3DP_predictions = joblib.load(results_dir + video_.split("/")[0] + ".pkl")
                else:               T3DP_predictions = joblib.load(results_dir + video_ + ".pkl")
            except: continue
            list_of_gt_frames = np.sort(list(data_gt[video_].keys()))
            tracked_frames    = list(T3DP_predictions.keys())
            data_all[video_]  = {}
            for i in range(len(list_of_gt_frames)):
                frame_        = list_of_gt_frames[i]
                total_annoated_frames += 1
                if(frame_ in tracked_frames):
                    tracked_data = T3DP_predictions[frame_]
                    if(len(data_gt[video_][frame_][0])>0):
                        assert data_gt[video_][frame_][0][0].split("/")[-1] == frame_
                        if(len(tracked_data[0])==0):   
                            data_all[video_][frame_] = [data_gt[video_][frame_][0], data_gt[video_][frame_][1], data_gt[video_][frame_][2], data_gt[video_][frame_][3], [], [], []]
                        else:
                            data_all[video_][frame_] = [data_gt[video_][frame_][0], data_gt[video_][frame_][1], data_gt[video_][frame_][2], data_gt[video_][frame_][3], frame_, tracked_data[0], tracked_data[1]] 
                            total_detected_frames += 1
                else:
                    data_all[video_][frame_] = [data_gt[video_][frame_][0], data_gt[video_][frame_][1], data_gt[video_][frame_][2], data_gt[video_][frame_][3], [], [], []]; 
     
    joblib.dump(data_all, results_dir + '/'+str(dataset)+'_'+str(method)+'.pkl')
        
        
        
    # #########################################################################################################
    # #########################################################################################################
    # ###############################             Evaluate             #######################################
    # #########################################################################################################
    # #########################################################################################################
  


    accumulators = []   
    for video in tqdm(list(data_all.keys())):
    
        acc = mm.MOTAccumulator(auto_id=True)
        accumulators.append(acc)
        
        for t, frame in enumerate(data_all[video].keys()):

            data = data_all[video][frame]

            gt_ids      = data[1]
            gt_ids_new  = data[3]
            gt_bbox     = data[2]
            pt_ids      = data[5]
            pt_bbox     = data[6]

            if(len(gt_ids_new)>0):
                cost_ = mm.distances.iou_matrix(gt_bbox, pt_bbox, max_iou=0.99)
                accumulators[-1].update(
                                                    gt_ids_new,  # Ground truth objects in this frame
                                                    pt_ids,      # Detector hypotheses in this frame
                                                    cost_
                        )

    mh = mm.metrics.create()

    summary = mh.compute_many(
        accumulators,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True
    )

    ID_switches = summary['num_switches']['OVERALL']
    MOTA        = summary['mota']['OVERALL']
    PRCN        = summary['precision']['OVERALL']
    RCLL        = summary['recall']['OVERALL']

    strsummary  = mm.io.render_summary(
        summary,
        formatters = mh.formatters,
        namemap    = mm.io.motchallenge_metric_names
    )

    results_ID_switches       = summary['num_switches']['OVERALL']
    results_mota              = summary['mota']['OVERALL']
    
    print(strsummary)
    print("ID switches  ", results_ID_switches)
    print("MOTA         ", results_mota)      
        
    text_file = open(results_dir + '/str_summary.txt', "w")
    n = text_file.write(strsummary)
    text_file.close()

    return strsummary, summary
    
            
        
if __name__ == '__main__':
    
    results_dir = str(sys.argv[1])
    method      = str(sys.argv[2])
    dataset     = str(sys.argv[3])
    strsummary, summary = evaluate_trackers(results_dir, method=method, dataset=dataset, make_video=0)
