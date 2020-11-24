#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:49:55 2020

@author: Tobias Engelhardt Rasmussen

Trying to classify the flat forehands and the backhands only using the depth 
videos.
"""
import cv2
import csv
import torch as tc
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from torch.nn import Linear, Conv2d, BatchNorm2d, AvgPool2d, MaxPool2d, Dropout2d, Dropout, BatchNorm1d
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot, mode_dot
from time import process_time

tl.set_backend('pytorch')
path = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS"
os.chdir(path)

# %% Loading the filenames and making file with them
"""
To read and write the names of the videos into a file in order to manually
allocate the time for the middle of the stroke.
"""
path = "/Users/Tobias/Desktop/"
os.chdir(path)
directory = "Data/VIDEO_Depth/forehand_flat/"
files = sorted(os.listdir(directory))
file = open('backhand_filenames.csv', 'w')
writer = csv.writer(file)
for filename in files: writer.writerow([filename])
file.close()

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
FRAME_RATE = 18
LENGTH = 1.5
def loadVideo(filename, middle= None, blackwhite= True):
    """
    Loads a video and returns a 4D tensor (frame, channel, height, width).
    """
    cap = cv2.VideoCapture(filename)
    if (middle == None):
        numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        firstFrame = 0
        lastFrame = numFrames
    else:
        numFrames = int(LENGTH * FRAME_RATE + 1)
        firstFrame = int((middle - LENGTH / 2) * FRAME_RATE)
        lastFrame = int(firstFrame + LENGTH * FRAME_RATE)
    ch = 1 if blackwhite else 3
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = tc.empty((ch,numFrames, height, width))
    framesLoaded = 0
    framesRead = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False or framesRead == lastFrame:
            break
        framesRead += 1
        if framesRead >= firstFrame:
            if blackwhite:
                frame = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
            else:
                frame = np.moveaxis(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),-1,0)
            frames[:,framesLoaded,:,:] = tc.tensor(frame)
            framesLoaded += 1
    cap.release()
    return frames

def loadShotType(shotType, directory, inputFile= None, dontLoadInds= None):
    """
    Loads all the videos of a directory and makes them into a big tensor. If the inputfile is given, the output will be
    a big tensor of shape (numVideos, channels, numFrames, height, width), otherwise the output will be a list of 4D
    tensors with different number of Frames.
    Inputfile contains the middle of the shot (time), dontLoadsInds are the videos that will not be loaded, due to
    potential problems.
    Shot types :
     - 0  Forehand flat
     - 1  Backhand
    Output:
        Tensor of dimension (numVideos, channels, numFrames, height, width) where the 3 first channels correspond to the
        RGB video, while the last channel is the black/white depth video.
    """
    shotTypes = {
        0: "forehand_flat/",
        1: "backhand/"
    }
    directoryRGB = directory + "VIDEO_RGB/" + shotTypes.get(shotType)
    directoryDep = directory + "VIDEO_Depth/" + shotTypes.get(shotType)
    filenamesRGB = sorted(os.listdir(directoryRGB))
    filenamesDep = sorted(os.listdir(directoryDep))

    if inputFile == None:
        if dontLoadInds != None:
            if (len(dontLoadInds) > 1):
                dontLoadInds = sorted(dontLoadInds, reverse= True)
            for ind in dontLoadInds:
                filenamesRGB.pop(ind)
                filenamesDep.pop(ind)
        allVideos = []
        for i in range(len(filenamesRGB)):
            thisRGB = loadVideo(directoryRGB + filenamesRGB[i], blackwhite= False)
            thisDep = loadVideo(directoryDep + filenamesDep[i], blackwhite= True)
            allVideos.append(tc.cat((thisRGB,thisDep),0))
        return allVideos
    else:
        rFile = open(inputFile, "r")
        reader = csv.reader(rFile, delimiter= ";")
        files = list(reader);
        if dontLoadInds != None:
            if (len(dontLoadInds) > 1):
                dontLoadInds = sorted(dontLoadInds, reverse= True)
            for ind in dontLoadInds:
                files.pop(ind)
                filenamesRGB.pop(ind)
                filenamesDep.pop(ind)
        numVideos = len(filenamesRGB)
        thisRGB = loadVideo(directoryRGB + filenamesRGB[0], float(files[0][1]), blackwhite= False)
        thisDep = loadVideo(directoryDep + filenamesDep[0], float(files[0][1]), blackwhite= True)
        thisVideo = tc.cat((thisRGB, thisDep),0)
        allVideos = tc.empty((numVideos, *thisVideo.shape))
        allVideos[0] = thisVideo
        for i in range(1,numVideos):
            middle = float(files[i][1])
            thisRGB = loadVideo(directoryRGB + filenamesRGB[i], middle= middle, blackwhite= False)
            thisDep = loadVideo(directoryDep + filenamesDep[i], middle= middle, blackwhite= True)
            allVideos[i] = tc.cat((thisRGB, thisDep), 0)
        return allVideos

# %% Loading the videos
t = process_time()
directory = "/Users/Tobias/Desktop/Data/"
# Forehands
inputForehand = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/forehand_filenames_adapted.csv"
forehands = loadShotType(0, directory, inputFile= inputForehand)
# Backhand
inputBackhand = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/backhand_filenames_adapted.csv"
backhands = loadShotType(1, directory, inputFile= inputBackhand)
print("Time to load: ",process_time()-t)

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

# %% Making into a big video with multiple tennis shots
out = cv2.VideoWriter('All_forehands.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 18, (640, 480))

for i in range(forehands.shape[0]):
    video = forehands[i]
    for j in range(video.shape[0]):
        frame = video[j][0]
        frame = cv2.cvtColor(frame.numpy(), cv2.COLOR_GRAY2RGB)
        out.write(frame)
out.release()
