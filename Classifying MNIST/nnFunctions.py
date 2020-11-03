#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:32:27 2020

@author: Tobias
"""
import matplotlib.pyplot as plt
import numpy as np
import torch as tc
from mnist import MNIST
from sklearn.metrics import accuracy_score

# For using pytorch nn framework
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def loadMNIST(p = 1/3, normalize = True):
    """ For loading the MNIST data set and making an instance of the data-class 
        p percent of the training data being validation """
    mndata = MNIST()

    x_train, y_train = mndata.load_training()
    x_test, y_test = mndata.load_testing()
    
    # Making into stack of images
    x_train, y_train = np.array(x_train).astype('float32'), np.array(y_train).astype('int32')
    x_test, y_test = np.array(x_test).astype('float32'), np.array(y_test).astype('int32')
    
    channels, rows, cols = 1, 28, 28  # 1 channel since BW and 28x28 pics
    x_train = x_train.reshape((-1,channels,rows,cols))
    x_test = x_test.reshape((-1,channels,rows,cols))
    changeInd = int(x_train.shape[0]*(1-p))
    
    return Data(x_train[:changeInd], y_train[:changeInd], x_train[changeInd:], y_train[changeInd:], x_test, y_test, normalize)


def showImage(img, label=""):
    """ For being able to plot the handwritten digits. 
        Either one by one, or a matrix of them """
    plt.figure()
    plt.imshow(np.reshape(img,newshape=(28,28)),vmin=tc.min(img),vmax=tc.max(img),cmap="gray")
    plt.axis('off')
    plt.title(label)
    

def showWrong(data, preds, labels):
    """ Shows wrong prediction images with true and guess labels 
        from predictions and true labels """
    wrongs = np.where((preds == labels) == False)[0]
    print(len(wrongs))
    for i in range(np.min((24,len(wrongs)))):
        showImage(data[wrongs[i]], label = "True: " + str(labels[wrongs[i]]) + " Guess: " + str(preds[wrongs[i]]))
    
    
def plotMany(img_L,B=10,H=10):
    """ B is how many pictures on the x-axis, and H is the y-axis """
    if (type(img_L) == tc.Tensor): img_L = img_L.numpy()
    plt.figure()
    nr = 0
    canvas = np.zeros((1,28*B))
    for i in range(H):
        temp = img_L[nr].reshape((28,28))
        nr+= 1
        for j in range(B-1):
            temp = np.concatenate((temp,img_L[nr].reshape((28,28))),axis = 1)
            nr+= 1
        canvas = np.concatenate((canvas, temp),axis = 0)
    plt.imshow(canvas[1:,:],vmin=np.min(img_L),vmax=np.max(img_L),cmap="gray")
    plt.axis('off')
    

class Data():
    """ The data-class to hold the different data splits """
    
    def __init__(self,x_train, y_train, x_val, y_val, x_test, y_test, normalize = True):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        if (normalize):
            self.x_train /= 255
            self.x_val /= 255
            self.x_test /= 255
        
    def subset(self, nTr, nVal, nTe):
        return Data(self.x_train[:nTr],self.y_train[:nTr],
                    self.x_val[:nVal], self.y_val[:nVal], 
                    self.x_test[:nTe], self.y_test[:nTe], normalize = False)
        
    def size(self):
        return self.x_train.shape[0], self.x_val.shape[0], self.x_test.shape[0]
        
    def __repr__(self):
        train, val, test = self.size()
        out = "This is a data set of: \n" + str(train) + " training samples, \n" 
        out = out + str(val) + " validation samples, and: \n" + str(test)
        out = out + " testing samples."
        return out


""" The training loop """
get_slice = lambda i, size: range(i * size, (i+1)*size)

def training(net, data, batch_size, num_epochs, optimizer, every = 1):

    criterion = nn.CrossEntropyLoss()
    
    num_samples_train, num_samples_valid, num_samples_test = data.size()
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
            print("Epoch %3i : Train Loss %f , Train acc %f, Valid acc %f" % (
            epoch, losses[-1], train_acc_cur, valid_acc_cur))
    epochs = np.arange(len(train_acc))
    plt.figure()
    plt.plot(epochs, train_acc, 'r', epochs, valid_acc, 'b')
    plt.legend(['Train accuracy','Validation accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # The testing accuracy
    test_preds = tc.max(net(Variable(tc.from_numpy(data.x_test))), 1)[1]
    print("---------------|o|----------------\nTesting accuracy on %3i samples: %f" %(num_samples_test, accuracy_score(test_preds.numpy(),data.y_test)))