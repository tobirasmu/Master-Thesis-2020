#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:49:55 2020

@author: Tobias Engelhardt Rasmussen

Trying to classify the flat forehands and the backhands only using the depth 
videos.
"""
# True if using the HPC cluster
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
from tools.visualizer import plotAccs
from tools.trainer import train_epoch, eval_epoch, get_variable
from tools.models import Net, Net2

tl.set_backend('pytorch')

# %% Loading the data
t = time()
directory = "/zhome/2a/c/108156/Data_MSc/" if HPC else "/home/tenra/PycharmProjects/Data Master/"
X, Y = tc.load(directory + "data.pt")

print("Took {:.2f} seconds to load the data".format(time() - t))

# %% Trying with a CNN to classify the tennis shots in the two groups

N, channels, frames, height, width = X.shape
nTrain = int(0.85 * N)

# Initializing the CNN
net = Net2(channels, frames, height, width)

# Converting to cuda if available
if tc.cuda.is_available():
    print("----  Network converted to CUDA  ----\n")
    net = net.cuda()

# %% Testing one forward push
test = X[0:2]
t = time()
out = net(get_variable(Variable(test)))
print("Time to complete 2 forward pushes was {:.2f} seconds with outputs\n {}\n".format(time() - t, out))

# %% Training functions using cross-validation since the amount of data is low
BATCH_SIZE = 20
NUM_FOLDS = 5
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
MOMENTUM = 0.7
WEIGHT_DECAY = 0.01


def train(X_train, y_train):
    train_loss, train_accs, val_accs = tc.empty(NUM_FOLDS, NUM_EPOCHS), \
                                       tc.empty(NUM_FOLDS, NUM_EPOCHS), \
                                       tc.empty(NUM_FOLDS, NUM_EPOCHS)
    kf = list(KFold(NUM_FOLDS).split(X_train))

    for i, (train_inds, val_inds) in enumerate(kf):
        print("{:-^60s}\n{:-^60s}\n{:-^60s}\n\n".format('', " FOLD {:3d}".format(i + 1), ''))
        this_net = Net2(channels, frames, height, width)
        if tc.cuda.is_available():
            this_net = this_net.cuda()
        optimizer = optim.SGD(this_net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        epoch, interrupted = 0, False
        while epoch < NUM_EPOCHS:
            print("{:-^60s}".format(" EPOCH {:3d} ".format(epoch + 1)))
            print("{: ^20}{: ^20}{: ^20}".format("Train Loss:", "Train acc.:", "Val acc.:"))
            try:
                train_loss[i, epoch], train_accs[i, epoch] = train_epoch(this_net, X_train[train_inds],
                                                                         y_train[train_inds], optimizer=optimizer,
                                                                         batch_size=BATCH_SIZE)
                val_accs[i, epoch] = eval_epoch(this_net, X_train[val_inds], y_train[val_inds])
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                interrupted = True
                break

            print("{: ^20.4f}{: ^20.4f}{: ^20.4f}".format(train_loss[i, epoch],
                                                          train_accs[i, epoch], val_accs[i, epoch]))
            epoch += 1
        if interrupted:
            break

    saveAt = "/zhome/2a/c/108156/Outputs/accuracies.png" if HPC else \
        "/home/tenra/PycharmProjects/Results/accuracies.png"
    train_accs = tc.mean(train_accs, dim=0)
    val_accs = tc.mean(val_accs, dim=0)
    plotAccs(train_accs, val_accs, saveName=saveAt)
    print("{:-^60}\nFinished".format(""))


print("{:-^60s}".format(" Training details "))
print("{: ^20}{: ^20}{: ^20}".format("Learning rate:", "Batch size:", "Number of folds"))
print("{: ^20.4f}{: ^20d}{: ^20d}\n{:-^60}\n".format(LEARNING_RATE, BATCH_SIZE, NUM_FOLDS, ''))

train(X[:nTrain], Y[:nTrain])
# tc.save(net.cpu().state_dict(), "/home/tenra/PycharmProjects/Results/Networks/trained_network.pt")
