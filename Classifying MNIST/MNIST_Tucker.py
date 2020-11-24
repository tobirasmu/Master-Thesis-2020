#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:41:30 2020

@author: Tobias
"""
# %% Loading the data and all the libraries

from nnFunctions import training, loadMNIST, Data, showImage, showWrong, plotMany
from time import time, process_time, process_time_ns
import numpy as np
from numpy.linalg import pinv, inv
import tensorly as tl
from copy import deepcopy
import torch as tc
from tensorly.decomposition import parafac, tucker, partial_tucker, matrix_product_state
from tensorly.tenalg import kronecker, multi_mode_dot, mode_dot
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as Fun
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn import Linear, Conv2d, BatchNorm2d, AvgPool2d, MaxPool2d, Dropout2d, Dropout, BatchNorm1d
from VBMF import EVBMF
tl.set_backend('pytorch')
fullData = loadMNIST()

# %% Defining the full network from the LeNet-5 architecture.
_, channels, width, height = fullData.x_train.shape

def conv_dim(dim, kernel, stride, padding):
    return int((dim - kernel + 2 * padding) / stride + 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # The convolutions
        self.conv1 = Conv2d(in_channels = channels, out_channels=6, kernel_size=5, padding=2, stride=1)
        dim1 = conv_dim(height, kernel = 5, padding = 2, stride= 1)
        dim1P = conv_dim(dim1, kernel = 2, padding = 0, stride = 2)
        self.conv2 = Conv2d(in_channels = 6, out_channels = 16, kernel_size=5, padding=0, stride=1)
        dim2 = conv_dim(dim1P, kernel = 5, padding = 0, stride=1)
        dim2P = conv_dim(dim2, kernel = 2, padding = 0, stride = 2)
        
        # The average pooling
        self.pool = AvgPool2d(kernel_size = 2, stride = 2, padding=0)
        
        # The dropout
        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout2d(0.2)
        
        self.lin_in_feats = 16 * (dim2P **2)
        # The linear layers
        self.l1 = Linear(in_features = self.lin_in_feats, out_features=120, bias = True)
        self.l2 = Linear(in_features = 120, out_features=84, bias = True)
        self.l_out = Linear(in_features = 84, out_features=10, bias = True)
    
    def forward(self,x):
        x = relu(self.conv1(x))
        x = self.pool(x)
            
        x = relu(self.conv2(x))
        x = self.pool(x)
        #x = self.dropout2(x)
            
        x = tc.flatten(x, 1)
            
        x = relu(self.l1(x))
        #x = self.dropout1(x)
        x = relu(self.l2(x))
        return softmax(relu(self.l_out(x)), dim = 1)
        
net = Net()
print(net)
x = np.random.normal(0,1,(5,1,28,28)).astype('float32')
out = net(Variable(tc.from_numpy(x)))
print(out)

# %% Training the network
data = fullData.subset(40000, 20000, 10000)

# %% Training functions
BATCH_SIZE = 128
NUM_EPOCHS = 60

criterion = nn.CrossEntropyLoss()

get_slice = lambda i, size: range(i * size, (i+1)*size)

def train_epoch(thisNet, X, y, optimizer, ):
    
    num_samples = X.shape[0]
    num_batches = num_samples // BATCH_SIZE
    losses = []
    targs, preds = [], []
    
    thisNet.train()
    for i in range(num_batches):
        if (i % (num_batches // 10) == 0):
            print("--", end='')
        # Sending the batch through the network
        slce = get_slice(i,BATCH_SIZE)
        X_batch = Variable(tc.from_numpy(X[slce]))
        output = thisNet(X_batch)  
        # The targets
        y_batch = Variable(tc.from_numpy(y[slce]).long())        
        # Computing the error and doing the step
        optimizer.zero_grad()
        batch_loss = criterion(output, y_batch)
        batch_loss.backward()
        optimizer.step()
        
        losses.append(batch_loss.data.numpy())
        predictions = tc.max(output, 1)[1]
        targs += list(y[slce])
        preds += list(predictions.data.numpy())
    return np.mean(losses), accuracy_score(targs, preds)

def eval_epoch(thisNet, X, y):
    num_samples = X.shape[0]
    num_batches = num_samples // BATCH_SIZE
    targs, preds = [], []
    
    thisNet.eval()
    for i in range(num_batches):
        if (i % (num_batches // 10) == 0):
            print("--", end='')
        slce = get_slice(i,BATCH_SIZE)
        X_batch_val = Variable(tc.from_numpy(X[slce]))
        output = thisNet(X_batch_val)
        
        predictions = tc.max(output, 1)[1]
        targs += list(y[slce])
        preds += list(predictions.data.numpy())
    return accuracy_score(targs, preds)

def train(thisNet, data, lr = 0.1, momentum = 0.5, factor = 1.1):
    train_accs, valid_accs, test_accs = [], [], []
    m_inc = (0.9 - momentum)/8
    inc = NUM_EPOCHS // 8 # We want 8 updates to the momentum and the learning rate
    epoch = 0
    while epoch < NUM_EPOCHS:
        epoch += 1
        try:
            if (epoch % inc == 0):
                momentum += m_inc
                momentum = momentum if momentum <= 0.9 else 0.9
                lr /= factor
            optimizer = optim.SGD(thisNet.parameters(), lr = lr, momentum=momentum)
            print("Epoch %d: (mom: %f, lr: %f)" % (epoch, momentum, lr))
            print("Train: ", end = '')
            loss, train_acc = train_epoch(thisNet, data.x_train, data.y_train, optimizer)
            print(" Validation: ", end = '')
            valid_acc = eval_epoch(thisNet, data.x_val, data.y_val)
            print(" Testing: ", end = '')
            test_acc = eval_epoch(thisNet, data.x_test, data.y_test)
            train_accs += [train_acc]
            valid_accs += [valid_acc]
            test_accs  += [test_acc]
            print("\nCost {0:.7}, train acc: {1:.4}, val acc: {2:.4}, test acc: {3:.4}". format(
                loss, train_acc, valid_acc, test_acc))
        except KeyboardInterrupt:
            print('\n KeyboardInterrupt')
            break
    epochs = np.arange(len(train_accs))
    plt.figure()
    plt.plot(epochs, train_accs, 'r', epochs, valid_accs, 'b')
    plt.legend(['Train accuracy','Validation accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

# %% Training the full network
train(net, data)
        
# %% Making the decomposed version
def conv_to_tucker2(layer, ranks= None):
    """
    Takes a pretrained convolutional layer and decomposes is using partial
    tucker with the given ranks.
    """
    # Making the decomposition of the weights
    weights =  layer.weight.data
    # (Estimating the ranks using VBMF)
    ranks = estimate_ranks(weights, [0,1]) if (ranks == None) else ranks
    # Decomposing
    core, [last, first] = partial_tucker(weights, modes = [0,1], ranks = ranks)
    
    # Making the layer into 3 sequential layers using the decomposition
    first_layer = Conv2d(in_channels=first.shape[0], out_channels=first.shape[1], 
                         kernel_size=1, stride=1, padding=0, bias = False)
    
    core_layer = Conv2d(in_channels=core.shape[1], out_channels=core.shape[0], 
                        kernel_size= layer.kernel_size, stride= layer.stride, 
                        padding= layer.padding, bias = False)
    
    last_layer = Conv2d(in_channels=last.shape[1], out_channels=last.shape[0], 
                        kernel_size=1, stride=1, padding=0, bias=True)
    
    # The decomposition is chosen as weights in the network (output, input, height, width)
    first_layer.weight.data = tc.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core # no reshaping needed
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    
    # The bias from the original layer is added to the last convolution
    last_layer.bias.data = layer.bias.data
    
    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)

def conv_to_tucker1(layer, rank= None):
    """
    Takes a pretrained convolutional layer and decomposes it using partial tucker with the given rank.
    """
    # Making the decomposition of the weights
    weights =  layer.weight.data
    out_ch, in_ch, kernel_size, _ = weights.shape
    # (Estimating the rank)
    rank = estimate_ranks(weights, [0]) if (rank == None) else [rank]
    core, [last] = partial_tucker(weights, modes = [0], ranks= rank)
    
    # Turning the convolutional layer into a sequence of two smaller convolutions
    core_layer = Conv2d(in_channels=in_ch, out_channels=rank[0], kernel_size=kernel_size, padding=layer.padding, 
                        stride= layer.stride, bias = False)
    last_layer = Conv2d(in_channels=rank[0], out_channels=out_ch, kernel_size= 1, padding=0, stride= 1, bias = True)
    
    # Setting the weights:
    core_layer.weight.data = core
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    last_layer.bias.data = layer.bias.data
    
    new_layers = [core_layer, last_layer]
    return nn.Sequential(*new_layers)

def lin_to_tucker1(layer, rank= None, in_channels= True):
    """
    Takes a linear layer as input, decomposes it using tucker1, and makes it into
    a sequence of two smaller linear layers using the decomposed weights. 
    """
    # Making the decomposition of the weights
    weights = layer.weight.data
    nOut, nIn = weights.shape
    if (in_channels):
        rank = estimate_ranks(weights, [1]) if (rank == None) else [rank]
        core, [B] = partial_tucker(weights, modes = [1], ranks= rank)
        
        # Now we have W = GB^T, we need Wb which means we can seperate into two layers
        BTb = Linear(in_features= nIn, out_features=rank[0], bias= False)
        coreBtb = Linear(in_features= rank[0], out_features=nOut, bias= True)
        
        # Set up the weights
        BTb.weight.data = tc.transpose(B, 0, 1)
        coreBtb.weight.data = core
        
        # Bias goes on last layer
        coreBtb.bias.data = layer.bias.data
        
        new_layers = [BTb, coreBtb]
    else:
        rank = estimate_ranks(weights, [0]) if (rank == None) else [rank]
        core, [A] = partial_tucker(weights, modes = [0], ranks= rank)
    
        # Now we have W = AG, we need Wb which means we can do Wb = A (Gb) as two linear layers
        coreb = Linear(in_features=nIn, out_features=rank[0], bias = False)    
        Acoreb = Linear(in_features=rank[0], out_features=nOut, bias = True)
    
        # Let the decomposed weights be the weights of the new
        coreb.weight.data = core
        Acoreb.weight.data = A
    
        # The bias goes on the second one
        Acoreb.bias.data = layer.bias.data
    
        new_layers = [coreb, Acoreb]
    return nn.Sequential(*new_layers)

def lin_to_tucker2(layer, ranks=None):
    """
    Takes in a linear layer and decomposes it by tucker-2. Then splits the linear
    map into a sequence of smaller linear maps. 
    """
    # Pulling out the weights
    weights = layer.weight.data
    nOut, nIn = weights.shape
    # Estimate (ranks and) weights
    ranks = estimate_ranks(weights, [0,1]) if (ranks == None) else ranks
    core, [A, B] = partial_tucker(weights, modes = [0,1], ranks = ranks)
    
    # Making the sequence of 3 smaller layers
    BTb = Linear(in_features=nIn, out_features=ranks[1], bias = False)
    coreBTb = Linear(in_features=ranks[1], out_features=ranks[0], bias = False)
    AcoreBTb = Linear(in_features=ranks[0], out_features=nOut, bias = True)
    
    # Setting the weights
    BTb.weight.data = tc.transpose(B, 0, 1)
    coreBTb.weight.data = core
    AcoreBTb.weight.data = A
    AcoreBTb.bias.data = layer.bias.data
    
    new_layers = [BTb, coreBTb, AcoreBTb]
    return nn.Sequential(*new_layers)
    

def numParams(net):
    return(sum(np.prod(p.size()) for p in net.parameters()))

def estimate_ranks(weight_tensor, dimensions):
    ranks = []
    for dim in dimensions:
        _, diag, _, _ = EVBMF(tl.unfold(weight_tensor, dim))
        ranks.append(diag.shape[dim])
    return ranks

# %% Trying to decompose the learned network
# Making a copy
netDec = deepcopy(net)

netDec.conv1 = conv_to_tucker1(netDec.conv1)
netDec.conv2 = conv_to_tucker2(netDec.conv2)
netDec.l1 = lin_to_tucker2(netDec.l1)
netDec.l2 = lin_to_tucker1(netDec.l2)

# %%
# The change in number of parameters
print("Before: {}, after: {}, which is {:.3}".format(numParams(net), numParams(netDec), numParams(netDec) / numParams(net)))
# Not the biggest decomp... How about accuracy?
t = time()
acc2 = eval_epoch(netDec, data.x_test, data.y_test)
print(time()-t)
t = time()
acc1 = eval_epoch(net, data.x_test, data.y_test)
print(time()-t)
print("\nAccuracy before: ",acc1)
print("\nAccuracy after: ",acc2)
# So the accuracy have decreased by a bit. Maybe it can be finetuned?

# %% Timing just one forward pass
x = Variable(tc.from_numpy(data.x_train[100]).unsqueeze(0))
allTimeNet = 0
allTimeDec = 0
for i in range(1000):
    t = process_time_ns()
    net(x)
    allTimeNet += process_time_ns()-t
    t = process_time_ns()
    netDec(x)
    allTimeDec += process_time_ns()-t
print(allTimeNet/1000, " ", allTimeDec/1000, "  ", (allTimeDec/1000)/(allTimeNet/1000))

# %% Fine-tuning the decomposed network
train(netDec, data, lr = 0.01, factor= 2)

# %% 
print("Before: {}, after: {}, which is {:.3}".format(numParams(net), numParams(netDec), numParams(netDec) / numParams(net)))
# Not the biggest decomp... How about accuracy?
times1 = []
for i in range(10):
    t = process_time()
    acc1 = eval_epoch(netDec, data.x_test, data.y_test)
    times1.append(process_time()-t)
print(np.mean(times1[1:]))
times2 = []
for i in range(10):
    t = process_time()
    acc2 = eval_epoch(net, data.x_test, data.y_test)
    times2.append(process_time()-t)
print(np.mean(times2[1:]))
print("\nAccuracy before: ",acc2)
print("\nAccuracy after: ",acc1)

