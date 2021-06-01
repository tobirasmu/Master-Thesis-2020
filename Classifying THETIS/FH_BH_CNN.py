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
HPC = False
path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/"
os.chdir(path)

from time import process_time
import torch as tc
import torch.optim as optim
import tensorly as tl
from torch.autograd import Variable
from sklearn.model_selection import KFold
from video_functions import get_variable, plotAccs, train_epoch, eval_epoch

tl.set_backend('pytorch')

# %% Loading the data
t = process_time()
directory = "/zhome/2a/c/108156/Data_MSc/" if HPC else "/Users/Tobias/Desktop/Data/"
X, Y = tc.load(directory + "data.pt")

print("Took {:.2f} seconds to load the data".format(process_time() - t))

# %% Trying with a CNN to classify the tennis shots in the two groups
from video_networks import Net

N, channels, frames, height, width = X.shape
nTrain = int(0.85 * N)

# Initializing the CNN
net = Net(channels, frames, height, width)

# Converting to cuda if available
if tc.cuda.is_available():
    print("----  Network converted to CUDA  ----\n")
    net = net.cuda()
# %% Testing one forward push
test = X[0:2]
t = process_time()
out = net(get_variable(Variable(test)))
print("Time to complete 2 forward pushes was {:.2f} seconds with outputs\n {}\n".format(process_time() - t, out))

# %% Training functions using cross-validation since the amount of data is low
BATCH_SIZE = 10
NUM_FOLDS = 5
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.5, weight_decay=0.01)


def train(this_net, X_train, y_train, X_test, y_test):
    train_accs, val_accs, test_accs = tc.empty(NUM_EPOCHS), tc.empty(NUM_EPOCHS), tc.empty(NUM_EPOCHS)
    kf = list(KFold(NUM_FOLDS).split(X_train))
    epoch, interrupted = 0, False
    while epoch < NUM_EPOCHS:
        epoch += 1
        print("{:-^60s}".format(" EPOCH {:3d} ".format(epoch)))
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
        print("{: ^15}{: ^15}{: ^15}{: ^15}".format("Loss:", "Train acc.:", "Val acc.:", "Test acc.:"))
        print("{: ^15.4f}{: ^15.4f}{: ^15.4f}{: ^15.4f}".format(this_loss, this_train_acc, this_val_acc,
                                                                test_accs[epoch - 1]))
    saveAt = "/zhome/2a/c/108156/Outputs/accuracies.png" if HPC else "/Users/Tobias/Desktop/accuracies.png"
    plotAccs(train_accs, val_accs, saveName=saveAt)
    print("{:-^60}\nFinished".format(""))


print("{:-^60s}".format(" Training details "))
print("{: ^20}{: ^20}{: ^20}".format("Learning rate:", "Batch size:", "Number of folds"))
print("{: ^20.4f}{: ^20d}{: ^20d}\n{:-^60}".format(LEARNING_RATE, BATCH_SIZE, NUM_FOLDS, ''))

train(net, X[:nTrain], Y[:nTrain], X[nTrain:], Y[nTrain:])
tc.save(net.cpu().state_dict(), "/zhome/2a/c/108156/Outputs/trained_network.pt")
