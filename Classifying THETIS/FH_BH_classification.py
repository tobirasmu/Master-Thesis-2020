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
import numpy as np
import os
from sys import getsizeof
from tensorly.decomposition import partial_tucker, tucker
from tensorly.tenalg import multi_mode_dot
import tensorly as tl
tl.set_backend('pytorch')
import matplotlib.pyplot as plt

# %% Loading the filenames and making file with them

path = "/Users/Tobias/Desktop/"
os.chdir(path)
#directory = "Data/VIDEO_Depth/forehand_flat/"
directory = "Data/VIDEO_Depth/backhand/"
files = sorted(os.listdir(directory))
file = open('backhand_filenames.csv','w')
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

"""

# %% Forehands
LENGTH = 2.0
forehandsL = []
lengths = []
directory = "Data/VIDEO_Depth/forehand_flat/"
rFile = open("forehand_filenames_adapted.csv","r")
reader = csv.reader(rFile, delimiter = ";")
filenames = list(reader)

for filename in filenames:
    name = filename[0]
    middle = float(filename[1])
    
    videoFile = directory + name
    cap = cv2.VideoCapture(videoFile)
    firstFrame = int((middle-LENGTH/2)*18.0)
    lastFrame = int(firstFrame + LENGTH*18.0)
    frames = []
    numFrames = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if (ret != True or numFrames == lastFrame):
            break
        numFrames += 1
        if (numFrames >= firstFrame):
            # Converting to BW to minimize memory usage and speed learning
            frames.append(tc.from_numpy(np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), axis = 0)))
    forehandsL.append(tc.stack(frames))
    lengths.append(len(frames))
    
# Taking out the ones that cannot be used
removeInds = list(tc.where(tc.tensor(lengths) != int(2.0*18+1))[0]) + [10, 45]
removeInds.sort(reverse=True)
for ind in removeInds:
    forehandsL.pop(ind)
    
# Making into a big tensor
forehands = tc.stack(forehandsL)

# %% Making into a big video with lots of tennis shots
out = cv2.VideoWriter('All_forehands.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 18, (640, 480))

for i in range(forehands.shape[0]):
    video = forehands[i]
    for j in range(video.shape[0]):
        frame = video[j][0]
        frame = cv2.cvtColor(frame.numpy(), cv2.COLOR_GRAY2RGB)
        out.write(frame)
out.release()

# %% 
lengths2 = []
files = sorted(os.listdir("Data/VIDEO_RGB/backhand/"))
for filename in files:
    cap = cv2.VideoCapture("Data/VIDEO_RGB/backhand/" + filename)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if (ret != True):
            break
        frames.append(1)
    lengths2.append(len(frames))
        