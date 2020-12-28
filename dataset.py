#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import cv2
import math
from skimage.exposure import adjust_gamma
from image import *

def generate_heatmap(heatmap, sigma):
    heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
    am = np.amax(heatmap)
    if am == 0:
        print("this is a break")
    heatmap /= am / 255
    return heatmap

class listTxTDataset(Dataset):
    def __init__(self, imgdirpath, shape=None,
                 transform=None, target_transform=None):

        self.imgList = os.listdir(imgdirpath)
        self.nSamples = 0
        self.transform = transform
        self.target_transform = target_transform
        self.shape = shape
        self.imgdirpath = imgdirpath
        self.data = []
        self.train_list = []
        for f in os.listdir(imgdirpath):
            if '.npy' in f:
                continue
            else:
                self.train_list.append(f)

        self.nSamples = len(self.train_list)
        print("training samples:", self.nSamples)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= self.nSamples, 'index range error'
        image_id = self.train_list[index]
        image = cv2.imread(os.path.join(self.imgdirpath, image_id))
        anno = np.load(os.path.join(self.imgdirpath, image_id+'.npy'))

        no_box = anno.shape[0]//4
        s = image.shape
        loc = []
        for i in range(no_box):
            # (center (x, y), (width, height), angle of rotation ).
            tmp = anno[i*4:(i*4+4),:]
            c = np.mean(tmp, axis=0)
            v_w = tmp[2,:] - tmp[0,:]
            v_h = tmp[3, :] - tmp[1, :]
            w = np.linalg.norm(tmp[0,:] - tmp[2,:])
            h = np.linalg.norm(tmp[1, :] - tmp[3, :])
            angle = np.arccos(np.dot(v_h, v_w)/(w*h))
            loc.append([c[0]/s[1], c[1]/s[0], w/s[1], h/s[0], angle/np.pi])
        image = image[:,:,0] / 255.0
        image = cv2.resize(image, self.shape)

        loc = torch.tensor(np.array(loc)).float()
        image = torch.tensor(image)[None,:,:].float()
        return image, loc




