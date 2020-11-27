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
from time import process_time
import matplotlib.pyplot as plt
import torch as tc
import torch.optim as optim
import tensorly as tl
import torch.nn as nn
from torch.nn.functional import relu, softmax
from torch.nn import Linear, Conv2d, Conv3d, BatchNorm2d, AvgPool2d, MaxPool2d, MaxPool3d, Dropout2d, Dropout, \
    BatchNorm1d
from torch.autograd import Variable
from sklearn.model_selection import KFold

from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot, mode_dot

from video_functions import loadShotType, get_variable, get_data, writeNames2file, writeTensor2video, showFrame, \
                            plotAccs, train_epoch, eval_epoch

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
RESOLUTION = 0.25

t = process_time()
directory = "/Users/Tobias/Desktop/Data/"
# Forehands
inputForehand = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying " \
                "THETIS/forehand_filenames_adapted.csv "
forehands = loadShotType(0, directory, input_file=inputForehand, length=LENGTH, resolution=RESOLUTION)
# Backhand
inputBackhand = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying " \
                "THETIS/backhand_filenames_adapted.csv "
backhands = loadShotType(1, directory, input_file=inputBackhand, length=LENGTH, resolution=RESOLUTION)
print("Time to load: ", process_time() - t)

# %% Compile into one large dataset.
t = process_time()
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

print("Time to concatenate: ", process_time() - t)
print("X is a tensor of shape: ", *X.shape, " (num_obs ch frames height width)")

# %% Trying with a CNN to classify the tennis shots in the two groups
_, channels, frames, height, width = X.shape


def conv_dims(dims, kernels, strides, paddings):
    dimensions = len(dims)
    new_dims = tc.empty(dimensions)
    for i in range(dimensions):
        new_dims[i] = int((dims[i] - kernels[i] + 2 * paddings[i]) / strides[i] + 1)
    return new_dims


# First convolution
c1_channels = (channels, 6)
c1_kernel = (5, 11, 11)
c1_stride = (1, 1, 1)
c1_padding = (2, 5, 5)
# Second convolution
c2_channels = (6, 16)
c2_kernel = (5, 11, 11)
c2_stride = (1, 1, 1)
c2_padding = (0, 0, 0)
# Pooling layer
pool_kernel = (2, 4, 4)
pool_stride = (2, 4, 4)
pool_padding = (0, 0, 0)
# Linear layers
l1_features = 120
l2_features = 84
l_out_features = 2


# The CNN for the THETIS dataset
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Adding the convolutional layers
        self.c1 = Conv3d(in_channels=c1_channels[0], out_channels=c1_channels[1], kernel_size=c1_kernel,
                         stride=c1_stride, padding=c1_padding)
        dim1s = conv_dims((frames, height, width), kernels=c1_kernel, strides=c1_stride, paddings=c1_padding)
        dim1sP = conv_dims(dim1s, kernels=pool_kernel, strides=pool_stride, paddings=pool_padding)

        self.c2 = Conv3d(in_channels=c2_channels[0], out_channels=c2_channels[1], kernel_size=c2_kernel,
                         stride=c2_stride, padding=c2_padding)
        dim2s = conv_dims(dim1sP, kernels=c2_kernel, strides=c2_stride, paddings=c2_padding)
        dim2sP = conv_dims(dim2s, kernels=pool_kernel, strides=pool_stride, paddings=pool_padding)

        # The pooling layer
        self.pool3d = MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)

        # Features into the linear layers
        self.lin_feats_in = int(16 * tc.prod(dim2sP))
        # Adding the linear layers
        self.l1 = Linear(in_features=self.lin_feats_in, out_features=l1_features)
        self.l2 = Linear(in_features=l1_features, out_features=l2_features)
        self.l_out = Linear(in_features=l2_features, out_features=l_out_features)

    def forward(self, x):
        x = relu(self.c1(x))
        x = self.pool3d(x)

        x = relu(self.c2(x))
        x = self.pool3d(x)

        x = tc.flatten(x, 1)

        x = relu(self.l1(x))
        x = relu(self.l2(x))
        return softmax(self.l_out(x), dim=1)


# Initializing the CNN
net = Net()

# Converting to cuda if available
if tc.cuda.is_available():
    print("----  Network converted to CUDA  ----\n")
    net = net.cuda()
# %% Testing one forward push
test = X[0:2]
t = process_time()
out = net(Variable(test))
print("It took: ", process_time() - t, " seconds to complete a 2 forward push")
print(out)

# %% Training functions using cross-validation since the amount of data is low
BATCH_SIZE = 1
NUM_FOLDS = 2
NUM_EPOCHS = 2

optimizer = optim.Adam(net.parameters(), lr=0.001)


def train(this_net, X_train, y_train, X_test, y_test):
    train_accs, val_accs, test_accs = tc.empty(NUM_EPOCHS), tc.empty(NUM_EPOCHS), tc.empty(NUM_EPOCHS)
    kf = list(KFold(NUM_FOLDS).split(X_train))
    epoch, interrupted = 0, False
    while epoch < NUM_EPOCHS:
        epoch += 1
        print("{:->28} {:3d} {:-<22}".format(' EPOCH', epoch, ''))
        fold_loss = tc.empty(NUM_FOLDS)
        fold_train_accs = tc.empty(NUM_FOLDS)
        fold_val_accs = tc.empty(NUM_FOLDS)
        for i, (train_inds, val_inds) in enumerate(kf):
            try:
                fold_loss[i], fold_train_accs[i] = train_epoch(this_net, X_train[train_inds], y_train[train_inds],
                                                               optimizer=optimizer, batch_size=BATCH_SIZE)
                fold_val_accs[i] = eval_epoch(this_net, X_train[val_inds], y_train[val_inds])
            except KeyboardInterrupt:
                print('\nKeyboardInterrupt')
                interrupted = True
                break
        if interrupted is True:
            break
        this_loss, this_train_acc, this_val_acc = tc.mean(fold_loss), tc.mean(fold_train_accs), tc.mean(fold_val_accs)
        train_accs[epoch - 1], val_accs[epoch - 1] = this_train_acc, this_val_acc
        # Doing the testing evaluation
        test_accs[epoch - 1] = eval_epoch(this_net, X_test, y_test)
        print("{: >7}{: >16}{: >14}{: >15}".format("Loss:", "Train acc.:", "Val acc.:", "Test acc.:"))
        print("{0: 8.4f}{1: 13.4f}{2: 14.4f}{3: 15.4f}".format(this_loss, this_train_acc, this_val_acc,
                                                               test_accs[epoch - 1]))
    plotAccs(train_accs, val_accs)


train(net, X[:4], Y[:4], X[4:], Y[4:])
