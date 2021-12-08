from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# from dataloader import get_dataloaders

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
    

def normalize(x, dim=-1):
    norm1 = x / np.linalg.norm(x, axis=dim, keepdims=True)
    return norm1

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

