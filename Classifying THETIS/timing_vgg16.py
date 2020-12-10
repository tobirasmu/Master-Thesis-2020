#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:08:39 2020

@author: s153057
"""
import timeit
code = """
HPC = True

import os
path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/"
os.chdir(path)
import torchvision.models as models
from copy import deepcopy
import torch as tc
from torch.autograd import Variable
import numpy as np
import tensorly as tl
tl.set_backend('pytorch')
import cv2
from video_functions import conv_to_tucker1, conv_to_tucker2, lin_to_tucker1, lin_to_tucker2, numFLOPsPerPush, numParams, get_variable

directory = "/zhome/2a/c/108156/Data_MSc/" if HPC else "/Users/Tobias/Desktop/Data/"
vgg16 = models.vgg16(pretrained=True)

vgg16_dec = deepcopy(vgg16)
vgg16_dec.features[0] = conv_to_tucker1(vgg16.features[0])
vgg16_dec.features[2] = conv_to_tucker2(vgg16.features[2])
vgg16_dec.features[5] = conv_to_tucker2(vgg16.features[5])
vgg16_dec.features[7] = conv_to_tucker2(vgg16.features[7])
vgg16_dec.features[10] = conv_to_tucker2(vgg16.features[10])
vgg16_dec.features[12] = conv_to_tucker2(vgg16.features[12])
vgg16_dec.features[14] = conv_to_tucker2(vgg16.features[14])
vgg16_dec.features[17] = conv_to_tucker2(vgg16.features[17])
vgg16_dec.features[19] = conv_to_tucker2(vgg16.features[19])
vgg16_dec.features[21] = conv_to_tucker2(vgg16.features[21])
vgg16_dec.features[24] = conv_to_tucker2(vgg16.features[24])
vgg16_dec.features[26] = conv_to_tucker2(vgg16.features[26])
vgg16_dec.features[28] = conv_to_tucker2(vgg16.features[28])
#vgg16_dec.classifier[0] = lin_to_tucker2(vgg16.classifier[0])   # Takes LONG to decompose
vgg16_dec.classifier[3] = lin_to_tucker1(vgg16.classifier[3])
vgg16_dec.classifier[6] = lin_to_tucker1(vgg16.classifier[6])

print("{:-^60}{:-^60}{:-^60}".format('', " The VGG-16 network ", ''))
#Converting to cuda if possible
if tc.cuda.is_available():
    vgg16 = vgg16.cuda()
    vgg16_dec = vgg16_dec.cuda()
    print("-- Using GPU --")

print("Number of parameters:{: <30s}{: >12d}{: <30s}{: >12d}{: <30s}{: >12f}".format("Original:",numParams(vgg16),"Decomposed:", numParams(vgg16_dec), "Ratio:",numParams(vgg16_dec) / numParams(vgg16)))
# %% Calculating the theoretical and actual speed-ups
FLOPs_vgg16 = numFLOPsPerPush(vgg16, (224, 224), paddings=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], pooling=[2, 4, 7, 10, 13], pool_kernels=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
# Distribution of time used:
FLOPs_vgg16_dcmp = numFLOPsPerPush(vgg16_dec, (224, 224), paddings=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37], pooling=[5, 11, 20, 29, 38], pool_kernels=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])

# Overall theoretical speed-up
print("Overall ratio is {:.4f} hence the speed-up should be of around {:.2f} times".format(sum(FLOPs_vgg16_dcmp) / sum(FLOPs_vgg16), sum(FLOPs_vgg16) / sum(FLOPs_vgg16_dcmp)))

# Actual speed-up
test_cat = tc.tensor(np.moveaxis(cv2.cvtColor(cv2.imread(directory + "cat.png"), cv2.COLOR_BGR2RGB), -1, 0), dtype=tc.float).unsqueeze(0) / 255
test_ball = tc.tensor(np.moveaxis(cv2.cvtColor(cv2.imread(directory + "ball.png"), cv2.COLOR_BGR2RGB), -1, 0), dtype=tc.float).unsqueeze(0) / 255
test_vgg16 = Variable(get_variable(tc.cat((test_cat, test_ball), 0)))
"""

# %%
timeOrig = timeit.timeit("vgg16(test_vgg16)", setup=code, number=10)
timeNew = timeit.timeit("vgg16_dec(test_vgg16)", setup=code, number=10)

print("Actual speed-up was {} times based on {} seconds for the original and {} for the decomposed.".format(timeOrig / timeNew, timeOrig/20, timeNew/20))