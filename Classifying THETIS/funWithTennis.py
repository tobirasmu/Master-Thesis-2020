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