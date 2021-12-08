import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageDraw, ImageFont, ImageColor
from torch.optim.lr_scheduler import MultiStepLR

import random
import wandb
from utils.losses import ReIDLoss
from utils.utils_measure import AverageMeter

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

from test_t3dp import test_tracker

from models.hmar import HMAR
from utils.utils_measure import AverageMeter
from utils.make_video import refine_visuals, make_video

from HMAR_tracker import HMAR_tracker
from evaluate_t3dp import evaluate_trackers

from deep_sort_ import nn_matching
from deep_sort_.detection import Detection
from deep_sort_.tracker import Tracker
from dataset_posetrack import PoseTrack




def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--train', action='store_true', help='traning or testing')
    parser.add_argument('--test_before', action='store_true', help='traning or testing')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='4,8', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--tags', type=str, default="tracking,feb22", help='add tags for the experiment')

    # dataset
    parser.add_argument('--model', type=str, default='conv')
    parser.add_argument('--train_dataset', type=str, default='posetrack2018')
    parser.add_argument('--test_dataset', type=str, default='posetrack2018')
    parser.add_argument('--feature', type=str, default='A')
    parser.add_argument('--train_batch_size', type=int, default=3, help='number of train batches')
    parser.add_argument('--test_batch_size', type=int, default=1, help='number of test batches')

    # specify folder
    parser.add_argument('--model_path', type=str, default='save/', help='path to save model')
    parser.add_argument('--data_root', type=str, default='../../Datasets/MNIST', help='path to data root')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
        
    tags = opt.tags.split(',')
    opt.tags = list([])
    for it in tags:
        opt.tags.append(it)

    opt.model_name = '{}_{}'.format(opt.model, opt.train_dataset)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_gpu = torch.cuda.device_count()
    
    
    #extras
    opt.fresh_start = True
    
    return opt

    
def train(opt, hmar_tracker, train_data_loader, optimizer, scheduler):

    reid_loss          = ReIDLoss(cosine=False)      
    reid_loss          = reid_loss.cuda()
    best_IDs, IDs      = 10000, 0
    step_              = 0
    number_of_epochs   = opt.epochs//len(train_data_loader)
    
    for epoch in range(number_of_epochs):
        
        
        train_loss_reid       = AverageMeter()
        train_loss            = AverageMeter()

        with tqdm(train_data_loader, total=len(train_data_loader)) as pbar:
            for batch_idx, data  in enumerate(pbar):
                
                step_ = epoch*len(train_data_loader) + batch_idx
                scheduler.step()
                
                _, ids, centers, scales, bboxs, img_names, pose_emb, apper_emb, gt_keypoint, smpl = data
                    
                BS, T, P    = ids.size()
                output, _   = hmar_tracker.forward(BS, T, P, [pose_emb.cuda(), apper_emb.cuda(), smpl.cuda()], ids, bboxs, gt_keypoint)
                
                embeddings  = output["output_embeddings"] 
                ids_x       = output["ids"] 
                    
                re_id_loss_ = torch.tensor(0).float().cuda() * 0.0

                for bs_ in range(BS):
                    re_id_loss_  += reid_loss(ids_x[bs_], embeddings[bs_], T, P)

                re_id_loss_ /= BS           
                loss = re_id_loss_
                try:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except:
                    print("no gardients!")
                    


                train_loss_reid.update(re_id_loss_.item(), BS)
                train_loss.update(loss.item(), BS)
                
                pbar.set_postfix({"total_loss"   :'{0:.4f}'.format(loss.detach().cpu().numpy(), 2) })
            
                if((step_+399)%400==0):
                    hmar_tracker.eval()
                    x = opt.dataset
                    opt.dataset        = "posetrack"
                    opt.dataset_path   = "_DATA/Posetrack_2018/"
                    opt.th_x           = 20000
                    opt.past_x         = 20
                    opt.max_age_x      = 20
                    opt.n_init_x       = 5
                    opt.max_ids_x      = 50
                    opt.window_x       = 1
                    opt.metric_x       = "euclidean_min"
                    opt.render         = False
                    opt.save           = False
                    opt.downsample     = 1
                    opt.videos_seq     = np.load("_DATA/posetrack.npy")
    
                    test_tracker(opt, hmar_tracker)
                    opt.dataset        = x
                    hmar_tracker.train()
                    
                    _, summary         = evaluate_trackers("out/" + opt.storage_folder + "/results/", method="t3dp", dataset="posetrack")
                    IDs                = summary['num_switches']['OVERALL']
                    MOTA               = summary['mota']['OVERALL']
                    IDF1               = summary['idf1']['OVERALL']

                    wandb.log({
                        'step' : step_, 
                        'epoch': epoch, 
                        'IDs'  : IDs, 
                        'IDF1' : IDF1, 
                        'MOTA' : MOTA, 
                        'Total loss': train_loss.avg,
                        'ReID loss': train_loss_reid.avg,
                        "Learning Rate"  : scheduler.get_lr()[0]
                    })

                    print(wandb.run.name, "\t\t ID switches ",  IDs)
                    print(wandb.run.name, "\t\t MOTA ",  MOTA)

                    state = {
                        'epoch'     : epoch,
                        'optimizer' : optimizer.state_dict(),
                        'model'     : hmar_tracker.state_dict() if opt.n_gpu <= 1 else hmar_tracker.module.state_dict(),
                    }       
            
                    save_file = os.path.join(opt.save_folder, 'model_'+str(wandb.run.name)+'.pth')
                    torch.save(state, save_file)

                    if(IDs<best_IDs):
                        best_IDs = IDs
                        save_file = os.path.join(opt.save_folder, 'best_model_'+str(wandb.run.name)+'.pth')
                        torch.save(state, save_file)


        
        
        
if __name__ == '__main__':
    
    
    
    
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    
    seed = np.random.randint(10000) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

    
    opt      = parse_option()
    opt.seed = seed
    wandb.init(project="HMAR_tracking", tags=opt.tags, settings=wandb.Settings(start_method="fork"))
    wandb.config.update(opt)
    try: os.system("mkdir save")
    except: pass
    
    opt.wandb_name     = wandb.run.name
    opt.storage_folder = wandb.run.name
    opt.dataset        = "posetrack" 
      
    
    train_dataset      = PoseTrack(window = 5, frame_length = 20, img_size = 256, max_ids = 10)
    train_data_loader  = DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    hmar_tracker       = HMAR_tracker(mode=opt.feature, betas=[1.0,1.0,1.0])
    
    
    if torch.cuda.is_available():
        if opt.n_gpu > 1:
            hmar_tracker = nn.DataParallel(hmar_tracker)
            print("Number of GPUs : ", opt.n_gpu)
        hmar_tracker = hmar_tracker.cuda()
                
    
    optimizer          = optim.Adam(hmar_tracker.parameters(), lr=opt.learning_rate, weight_decay=5e-4)
    scheduler          = MultiStepLR(optimizer, milestones=opt.lr_decay_epochs, gamma=opt.lr_decay_rate)
    
    if(opt.train):
        train(opt, hmar_tracker, train_data_loader, optimizer, scheduler)
