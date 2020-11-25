#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:49:55 2020

@author: Tobias Engelhardt Rasmussen

Trying to classify the flat forehands and the backhands only using the depth 
videos.
"""
import os

path = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS"
os.chdir(path)
import cv2
from time import process_time
import matplotlib.pyplot as plt
import torch as tc

import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, AvgPool2d, MaxPool2d, Dropout2d, Dropout, BatchNorm1d
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot, mode_dot

from video_functions import loadShotType, write_names2file, write_tensor2video, showFrame

tl.set_backend('pytorch')

# %%

""" 
FOREHANDS
The file forehand_filenames has information about the middle of the stroke so
that the same amount of features can be extracted from each video.

Seems there are some problems with the depth videos for the following 2 instances:
There are not the same number of frames in the RGB and depth videos respectively.

    p13_foreflat_depth_s2.avi (10) - the one after is identical and works (11)
    p24_foreflat_depth_s1.avi (45)
   
    
BACKHANDS
The file backhand_filenames_adapted has information about the middle of the 
stroke.

The very first seems to be wrong. 
    p1_foreflat_depth_s1.avi (0) (RGB is wrong - is actually p50)
"""

# %% The loading functions
LENGTH = 1.5

t = process_time()
directory = "/Users/Tobias/Desktop/Data/"
# Forehands
inputForehand = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/forehand_filenames_adapted.csv"
forehands = loadShotType(0, directory, input_file=inputForehand, length=LENGTH)
# Backhand
inputBackhand = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/backhand_filenames_adapted.csv"
backhands = loadShotType(1, directory, input_file=inputBackhand, length=LENGTH)
print("Time to load: ", process_time() - t)

# %% Compile into one large dataset.
tc.manual_seed(42)
numForehands = forehands.shape[0]
numBackhands = backhands.shape[0]
N = numForehands + numBackhands
nTrain = int(0.8 * N)

X = tc.cat((forehands, backhands), dim=0)
del forehands, backhands
Y = tc.cat((tc.zeros(numForehands), tc.ones(numBackhands)))
permutation = tc.randperm(N)
X = X[permutation]
Y = Y[permutation]

# %% Trying to decompose the training tensor via Tucker
"""
Rank estimation does not work on a tensor that big since the required memory is 538 TiB (tebibyte), hence the ranks will
be chosen intuitively (obs, frame, height, width). We are interested in the temporal information hence, the frame dimension
will be given full rank (not decomposed). Since the frames are rather simple (BW depth), the spatial dimensions will not
be given full rank
"""
modes = [0, 2, 3]
ranks = [10, 120, 160]
core, [A, C, D] = partial_tucker(X[:nTrain, 0, :, :, :], modes=modes, ranks=ranks)


# Takes a significant amount of time

# %% Trying with a simple network to classify the tennis shots

class simpleNet(nn.Module):

    def __init__(self):
        super(simpleNet, self).__init__()

        # Adding the layers
        self.l1 = Linear(in_features=10, out_features=10, bias=True)
        self.l_out = Linear(in_features=10, out_features=2, bias=True)

    def forward(self, x):
        x = relu(self.l1(x))
        return softmax(self.l_out(x))


sNet = simpleNet()

# %% The approximation
wh = 50
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(X[wh, 0, 20, :, :], cmap='gray')
approximation = multi_mode_dot(core, [A[wh], C, D], modes=[0, 2, 3])
plt.subplot(1, 2, 2)
plt.imshow(approximation[20], cmap='gray')
plt.show()


# %% Showing a frame
import numpy as np
