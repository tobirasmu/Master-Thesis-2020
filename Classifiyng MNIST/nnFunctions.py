#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:32:27 2020

@author: Tobias
"""
import matplotlib.pyplot as plt
import numpy as np
import torch as tc
from sklearn.metrics import accuracy_score

# For using pytorch nn framework
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# For being able to plot the handwritten digits. Either one by one, or a matrix of them
def showImage(img, label=""):
    plt.imshow(np.reshape(img,newshape=(28,28)),vmin=0,vmax=255,cmap="gray")
    plt.title(label)
    
def plotMany(img_L,B=10,H=10):
    # B is how many pictures on the x-axis, and H is the y-axis
    nr = 0
    canvas = np.zeros((1,28*B))
    for i in range(H):
        temp = img_L[nr].reshape((28,28))
        nr+= 1
        for j in range(B-1):
            temp = np.concatenate((temp,img_L[nr].reshape((28,28))),axis = 1)
            nr+= 1
        canvas = np.concatenate((canvas, temp),axis = 0)
    plt.imshow(canvas[1:,:],vmin=0,vmax=255,cmap="gray")
    plt.axis('off')
    

# The data-class to hold the different data splits 
class Data():
    
    def __init__(self,x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        
    def size(self):
        return self.x_train.shape[0], self.x_val.shape[0]
        
    def __str__(self):
        out = "This is a data set of: \n" + str(self.x_train.shape[0]) + " training samples and \n" 
        out = out + str(self.x_val.shape[0]) + " validation samples."
        return out


# The training loop
get_slice = lambda i, size: range(i * size, (i+1)*size)

def training(net, data, batch_size, num_epochs, optimizer, every = 1):

    criterion = nn.CrossEntropyLoss()
    
    num_samples_train, num_samples_valid = data.size()
    num_batches_train = num_samples_train // batch_size
    num_batches_valid = num_samples_valid // batch_size
    
    # Setting up lists
    train_acc, train_loss = [], []
    valid_acc, valid_loss = [], []
    test_acc, test_loss = [], []
    cur_loss = 0
    losses = []
    
    for epoch in range(num_epochs):
        
        cur_loss = 0
        net.train()
        for i in range(num_batches_train):
            # Sending the batch throgh the network
            slce = get_slice(i,batch_size)
            x_batch = Variable(tc.from_numpy(data.x_train[slce]))
            output = net(x_batch)
            
            # Computing gradients and loss
            target_batch = Variable(tc.from_numpy(data.y_train[slce]).long())
            batch_loss = criterion(output, target_batch)
            optimizer. zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            # Updating the loss
            cur_loss += batch_loss
        losses.append(cur_loss / batch_size)
        
        net.eval()
        
        # Evaluating training data
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            x_batch = Variable(tc.from_numpy(data.x_train[slce]))
            
            output = net(x_batch)
            preds = tc.max(output, 1)[1]
            
            train_targs += list(data.y_train[slce])
            train_preds += list(preds.data.numpy())
        
        # Evaluating validation data
        valid_preds, valid_targs = [], []
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            x_batch = Variable(tc.from_numpy(data.x_val[slce]))
            
            output = net(x_batch)
            preds = tc.max(output, 1)[1]
            
            valid_targs += list(data.y_val[slce])
            valid_preds += list(preds.data.numpy())
        train_acc_cur = accuracy_score(train_targs, train_preds)
        valid_acc_cur = accuracy_score(valid_targs, valid_preds)
        
        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)
        
        if (epoch % (every) == 0):
            print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
            epoch, losses[-1], train_acc_cur, valid_acc_cur))
    epochs = np.arange(len(train_acc))
    plt.figure()
    plt.plot(epochs, train_acc, 'r', epochs, valid_acc, 'b')
    plt.legend(['Train accuracy','Validation accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')