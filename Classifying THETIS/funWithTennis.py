#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:03:37 2020

@author: Tobias
"""

import cv2
import torch as tc
import numpy as np
import os
from sys import getsizeof
from tensorly.decomposition import partial_tucker, tucker
from tensorly.tenalg import multi_mode_dot
import tensorly as tl
tl.set_backend('pytorch')
import matplotlib.pyplot as plt

        
# %% Getting just one video out

path = "/Users/Tobias/Desktop/"
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
os.chdir(path)
if (not os.path.exists('Test')):
    os.makedirs('Test')

videoFile = "/Users/Tobias/Desktop/Data/VIDEO_Mask/forehand_flat/p6_foreflat_Mask_s3.avi"
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path

out = cv2.VideoWriter('BWMask.avi',fourcc, 18, (640, 480),0)

frames = []
while(cap.isOpened()):
    frameId = cap.get(1)
    ret, frame = cap.read()
    if (not ret):
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(frame)
    out.write(frame)
out.release()

p6 = np.stack(frames)
#%%

for i in range(new.shape[0]):
    frame = errors[i]
    print(i, frame.shape)
    out.write(np.array(frame, dtype= np.uint8))
out.release()

# %%
coreHmm, [A, B, C] = tucker(tl.tensor(p6), rank = 2, ranks=(20,200,200))

# %%
p6 = tl.tensor(np.stack(frames))
# decomposing the frame to separate the dynamical information
core, [frames, W, H, ch] = partial_tucker(p6, modes = [0,1,2], ranks = [20,480, 640])

# %% New video
new = tl.tucker_to_tensor((core,[frames, W, H, ch]))
errors = p6-new

out = cv2.VideoWriter('ErrorsVideo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 18, (640, 480))

for i in range(new.shape[0]):
    frame = errors[i]
    out.write(np.array(frame, dtype= np.uint8))
out.release()

# %% Making all videos in a directory into tensors

amount = 0
if (not os.path.exists('Backhand_frames')):
    os.makedirs('Backhand_frames')
    
for root, dirs, files in os.walk("Data/VIDEO_RGB/backhand/."):
    
    for filename in files:
        amount += 1
        count = 0
        videoFile = "Data/VIDEO_RGB/backhand/" + filename
        cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        directory = "Backhand_frames/"
        
        frames = []
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            frames.append(frame)
        thisTensor = np.stack(frames)
        print(amount)
        np.save(directory + filename[:-4] + ".pt", thisTensor)
        cap.release()
