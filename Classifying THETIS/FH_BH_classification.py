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
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn import Linear, Conv2d, Conv3d, BatchNorm2d, AvgPool2d, MaxPool2d, MaxPool3d, Dropout2d, Dropout, BatchNorm1d
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot, mode_dot

from video_functions import loadShotType, writeNames2file, writeTensor2video, showFrame

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

print("X is a tensor of shape: ",*X.shape, " (num_obs ch frames height width)")

# %% Trying with a CNN to classify the tennis shots in the two groups
_, channels, frames, height, width = X.shape

def conv_dims(dims, kernels, strides, paddings):
    dimensions = len(dims)
    out = tc.empty(dimensions)
    for i in range(dimensions):
        out[i] = int((dims[i] - kernels[i] + 2 * paddings[i]) / strides[i] + 1)
    return out

c1_kernel = (5, 21, 21)
c1_stride = (1, 1, 1)
c1_padding = (2, 10, 10)

c2_kernel = (5, 21, 21)
c2_stride = (1, 1, 1)
c2_padding = (0, 0, 0)

pool_kernel = (2, 8, 8)
pool_stride = (2, 8, 8)
pool_padding = (0, 0, 0)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Adding the convolutional layers
        self.c1 = Conv3d(in_channels=channels, out_channels=16, kernel_size=c1_kernel, stride=c1_stride, padding=c1_padding)
        dim1s = conv_dims((frames, height, width), kernels=c1_kernel, strides=c1_stride, paddings=c1_padding)
        dim1sP = conv_dims(dim1s, kernels=pool_kernel, strides=pool_stride, paddings=pool_padding)

        self.c2 = Conv3d(in_channels=16, out_channels=64, kernel_size=(5, 21, 21), stride=1, padding=0)
        dim2s = conv_dims(dim1sP, kernels=c2_kernel, strides=c2_stride, paddings=c2_padding)
        dim2sP = conv_dims(dim2s, kernels=pool_kernel, strides=pool_stride, paddings=pool_padding)

        # The pooling layer
        self.pool3d = MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)

        # Features into the linear layers
        self.lin_feats_in = int(64 * tc.prod(dim2sP))
        # Adding the linear layers
        self.l1 = Linear(in_features=self.lin_feats_in, out_features=1000)
        self.l2 = Linear(in_features=1000, out_features=100)
        self.l_out = Linear(in_features=100, out_features=2)

    def forward(self, x):
        x = relu(self.c1(x))
        x = self.pool3d(x)

        x = relu(self.c2(x))
        x = self.pool3d(x)

        x = tc.flatten(x, 0)

        x = relu(self.l1(x))
        x = relu(self.l2(x))
        return softmax(self.l_out(x))


net = Net()

# %% Testing one forward push
test = X[0:2]
out = net(Variable(test))
# %% The approximation
wh = 50
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(X[wh, 0, 20, :, :], cmap='gray')
approximation = multi_mode_dot(core, [A[wh], C, D], modes=[0, 2, 3])
plt.subplot(1, 2, 2)
plt.imshow(approximation[20], cmap='gray')
plt.show()

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