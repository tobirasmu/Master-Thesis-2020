#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:09:15 2021

@author: tenra
"""

"""
The file "FH_BH_CNN.py" holds the K-fold corss validation for hyper parameter searching. Net2 was found to be optimal
"""

import os

HPC = True
path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/" if HPC else \
    "/home/tenra/PycharmProjects/Master-Thesis-2020/Classifying THETIS/"
os.chdir(path)

from time import time
import torch as tc
import torch.optim as optim
import tensorly as tl
from torch.autograd import Variable
from sklearn.model_selection import KFold
from tools.visualizer import plotAccs, plotFoldAccs
from tools.trainer import train_epoch, eval_epoch, get_variable
from tools.models import Net, Net2

tl.set_backend('pytorch')

# %% Loading the data
directory = "/zhome/2a/c/108156/Data_MSc/" if HPC else "/home/tenra/PycharmProjects/Data Master/"
X, Y = tc.load(directory + "data.pt")

# %% Defining the network
N, channels, frames, height, width = X.shape
nTrain = int(0.90 * N)

# Initializing the CNN
net = Net2(channels, frames, height, width)

# Converting to cuda if available
if tc.cuda.is_available():
    print("----  Network converted to CUDA  ----\n")
    net = net.cuda()
    
# %% Training the network
BATCH_SIZE = 20
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
MOMENTUM = 0.7
WEIGHT_DECAY = 0.01


def train(X_train, y_train):
    train_loss, train_accs = tc.empty(NUM_EPOCHS), tc.empty(NUM_EPOCHS)
    
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    epoch, interrupted = 0, False
    while epoch < NUM_EPOCHS:
        print("{:-^40s}".format(" EPOCH {:3d} ".format(epoch + 1)))
        print("{: ^20}{: ^20}".format("Train Loss:", "Train acc.:"))
        try:
            train_loss[epoch], train_accs[epoch] = train_epoch(net, X_train, y_train, optimizer=optimizer, 
                                                               batch_size=BATCH_SIZE)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt")
            interrupted = True
            break

        print("{: ^20.4f}{: ^20.4f}".format(train_loss[epoch], train_accs[epoch]))
        epoch += 1
        if interrupted:
            break
    saveAt = "/zhome/2a/c/108156/Outputs/accuracies.png" if HPC else \
        "/home/tenra/PycharmProjects/Results/accuracies.png"
    plotAccs(train_accs, saveName=saveAt)
    print("{:-^40}\n".format(""))
    print(f"{'Testing accuracy:':-^40}\n{eval_epoch(net, X[nTrain:], Y[nTrain:]): ^40.4f}")


if __name__=="__main__":
    train(X[:nTrain], Y[:nTrain])
    if HPC:
        tc.save(net.cpu().state_dict(), "/zhome/2a/c/108156/Outputs/trained_network.pt")
        