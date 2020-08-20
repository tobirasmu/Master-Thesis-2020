#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:59:51 2020

@author: Tobias

In the following an initial ANN and CNN will be made for the MNIST dataset. It
will be done using the pytorch-framework.

"""

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
import torch as tc

mndata = MNIST()

x_train, y_train = mndata.load_training()
x_test, y_test = mndata.load_testing()

x_train, y_train = np.array(x_train).astype('float32'), np.array(y_train).astype('int32')
x_test, y_test = np.array(x_test).astype('float32'), np.array(y_test).astype('int32')

#%% For being able to plot the handwritten digits. Either one by one, or a matrix of them
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

    #%%
    
# Example of plotting
plt.subplot(1,2,1)
showImage(x_train[201],y_train[201])
plt.subplot(1,2,2)
showImage(x_train[100],y_train[100])
#%%
# Some of the pictures are hard to recognize even for humans.
plotMany(x_train,30,20)
#%% ANN using the pytorch framework
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as Fun
import torch.optim as optim
import torch.nn.init as init

# Hyper parameters from this data
num_classes = 10
num_features = x_train.shape[1]
num_hidden = 512

#%%
    
# Defining the inital network
class Net(nn.Module):
    
    def __init__(self, num_features, num_hidden, num_output):
        super(Net,self).__init__() # Initializing and inheriting from the nn.module class
        
        # Input layer
        self.W_1 = Parameter(init.kaiming_uniform_(tc.Tensor(num_hidden,num_features)))
        self.b_1 = Parameter(init.constant_(tc.Tensor(num_hidden),0))
        
        # Hidden layer
        self.W_2 = Parameter(init.kaiming_uniform_(tc.Tensor(num_output,num_hidden)))
        self.b_2 = Parameter(init.constant_(tc.Tensor(num_output),0))
        
        # Activation
        self.activation = tc.nn.ELU()
        
    def forward(self,x):
        x = Fun.linear(x,self.W_1, self.b_1)
        x = self.activation(x)
        x = Fun.linear(x,self.W_2, self.b_2)
        return x
        
# Initializing an instance of the network

net = Net(num_features, num_hidden, num_classes)

#%% How to forward-pass some dummy data

x = np.random.normal(0, 1, (45, 28*28)).astype('float32')
B = net(Variable(tc.from_numpy(x))) # same as net.forward

#%% 
# Defining the inital network
class Net2(nn.Module):
    
    def __init__(self, num_features, num_hidden, num_hidden2, num_output):
        super(Net2,self).__init__() # Initializing and inheriting from the nn.module class
        
        # Input layer
        self.W_1 = Parameter(init.xavier_normal_(tc.Tensor(num_hidden,num_features)))
        self.b_1 = Parameter(init.constant_(tc.Tensor(num_hidden),0))
        
        # Hidden layer
        self.W_2 = Parameter(init.xavier_normal_(tc.Tensor(num_hidden2,num_hidden)))
        self.b_2 = Parameter(init.constant_(tc.Tensor(num_hidden2),0))
        
        # Hidden layer 2
        self.W_3 = Parameter(init.xavier_normal_(tc.Tensor(num_output,num_hidden2)))
        self.b_3 = Parameter(init.constant_(tc.Tensor(num_output),0))
        
        # Activation
        self.activation = tc.nn.Sigmoid()
        
    def forward(self,x):
        x = Fun.linear(x,self.W_1, self.b_1)
        x = self.activation(x)
        x = Fun.linear(x,self.W_2, self.b_2)
        x = self.activation(x)
        x = Fun.linear(x,self.W_3, self.b_3)
        return x
        
#%% Defining the data-class
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

# %% Defining the training loop function
from sklearn.metrics import accuracy_score

get_slice = lambda i, size: range(i * size, (i+1)*size)

def training(net, data, batch_size, num_epochs):
    
    optimizer = optim.SGD(net.parameters(), lr=0.1)
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
            target_batch = Variable(tc.from_numpy(y_train[slce]).long())
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
            
            train_targs += list(y_train[slce])
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
        
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
            epoch, losses[-1], train_acc_cur, valid_acc_cur))

# %% Testing with some data
        # Using 1000 samples and 500 validation samples
        
net = Net2(784, 512, 512, 10)
data = Data(x_train[:2000,:],y_train[:2000],x_train[2000:3000,:],y_train[2000:3000])
print(data)

training(net,data,100,100)
    
# %%

output = net(tc.from_numpy(x_train[3000:4000]))
preds = tc.max(output, 1)[1]

print(y_train[3000:4000]==preds.numpy())
    
    
    
    
    
    
    
    
    
    
    

