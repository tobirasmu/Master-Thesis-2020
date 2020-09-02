#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:23:08 2020

@author: Tobias

Tutorials from Andreas to understand the pytorch API

"""

# %% Understanding autograd(): Linear model implementation using Pytorch
# Set x, y, w and calculate loss, l
x = 2
y = 6
w = 17.5
learning_rate = 0.01
# Calculate Loss
import torch as tc

def get_loss(x, y, w):
    w = tc.tensor(w, dtype=tc.float, requires_grad=True)
    y_hat = y - x*w + 2
    loss = y_hat**2
    return w, loss

# Iteratively use Gradient Descent to find optimal w
for iteration in range(200):
    print(w)
    w, l = get_loss(x, y, w)
    l.backward() # Calculates the gradient of the variables related to the loss
    w = w - learning_rate*w.grad.data
    
# %% Having fun diferentiating a parabola
    
def FUN(x):
    x = tc.tensor(x, dtype=tc.float, requires_grad=True)
    return x, 4*x**2 - 5*x + 10

def Grad(x):
    return 8*x - 5

x, y = FUN(10.)
y.backward()
print(x.grad.data)

while (True):
    x, y = FUN(x)
    y.backward()
    gradient = x.grad.data
    x = x - 0.001*gradient
    if (gradient < 10**-4 and gradient > -10**-4):
        print ("Svaret er: ", x)
        break

# %% The function plottet - real minimum is 0.625
import matplotlib as plt
import numpy as np

xer = np.arange(-10,10,0.01)
hej, yer = FUN(xer)

plt.pyplot.plot(xer, yer.data.numpy())

# %% Very simple network with only one neuron
   
# This script illustrates how Pytorch can be used to implement Linear regression using Gradient Descent
import torch
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
class  Model(torch.nn.Module):
    def __init__(self):
        # In the constructor we instantiate two nn.Linear module
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
model = Model()
# Construct loss
criterion = torch.nn.MSELoss(size_average=False)
# Construct optimizer (Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Training loop
for epoch in range(1000):
    # Forward pass: Compute predicetd y by passing x to the model
    y_pred = model(x_data)
    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print('\nEpoch:', epoch, '\nloss:', loss.data, '\nparameters:', model.state_dict())
    # Set gradient to zero in each iteration, perform backward pass and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
# %% Using pytorch API to approximate a sine curve
    
import torch
from torch.nn import Module, Linear, MSELoss, ReLU, Sigmoid
from torch.optim import SGD
import numpy as np
from matplotlib import pyplot as plt
# Simple ANN with one hidden layer to approximate a sine curve
class ann(Module):
    def __init__(self):
        super(ann, self).__init__()
        self.l1 = Linear(1, 200)
        self.l2 = Linear(200, 1)
                
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = self.l2(x)
        return x

# l1 is initialized with random weights
my_ann = ann()
print(my_ann.state_dict())
# Forward propagation using the initial random weights
my_ann(torch.from_numpy(np.array([2])).float())
# Train lm1 to some data
torch.manual_seed(2019)
np.random.seed(2019)
num_samples = 800
x = np.random.uniform(1, 2*np.pi, num_samples)
y = np.random.normal(np.sin(x), 0.1)
x = torch.from_numpy(x).float().view(num_samples, -1)
y = torch.from_numpy(y).float().view(num_samples, -1)
criterion = MSELoss()
optimizer = SGD(my_ann.parameters(), lr=0.005)
for epoch in range(50000):
    pred = my_ann(x)
    loss = criterion(y, pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%1000 == 0:
        print('\nepoch:', epoch, '\nloss:', loss)
print('\nfinal weights:', my_ann.state_dict())
#Plot results
def get_pred_data(x, model):
    x_pred = torch.linspace(x.min(), x.max(), 200).view(200, -1)
    #Can't call numpy() on Variable that requires grad so var.detach().numpy() is used
    y_pred = model(x_pred).detach().numpy()[:,0]
    x_pred = x_pred.detach().numpy()[:,0]
    return x_pred, y_pred

plt.scatter(x, y)
x_pred, y_pred = get_pred_data(x, my_ann)
plt.plot(x_pred, y_pred, color='red')
plt.show()
