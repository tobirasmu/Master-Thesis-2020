#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:38:38 2020

@author: Tobias

In this file, it is attempted to decompose some 3s and 4s from the MNIST data set.
"""

from nnFunctions import training, loadMNIST, Data, showImage, showWrong, plotMany, get_slice
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker, non_negative_parafac, non_negative_tucker
import matplotlib.pyplot as plt

data = loadMNIST()

# %% Making a tensor of only 3s and 4s. 

X_all = data.x_train
Y_all = data.y_train

indices4_and_3 = np.where(((data.y_train == 4) + (data.y_train == 3)))

X = X_all[indices4_and_3]
Y = Y_all[indices4_and_3]

plotMany(X, 30, 20)

""" 
Now X is a tensor of only 3s and 4s
"""

# %% 

K = tucker(tl.tensor(X[:1000].reshape((-1,28,28))), ranks=[2,5,5])
loadings = K[1]
core = K[0]

X_hat = tl.tucker_to_tensor(K)

plt.close('all')
plotMany(X, 20, 10)
plotMany(X_hat, 10, 1)




from tensorly.tenalg import multi_mode_dot
G0 = multi_mode_dot(core[1], [loadings[1], loadings[2]], modes = [0,1])
#plt.figure()
showImage(G0)
#plt.figure()

# %% TT-decomp first looks
r = (1,4,9,1)

G1 = tl.tensor([[1,4,5,6,1,3,5]])
G2 = tl.tensor([[1,2,3],[3,2,5],[4,5,6],[8,6,3],[1,4,2],[1,2,3],[2,3,4]])
G3 = tl.tensor([[8,3,1]]).reshape((3,1))
hej = np.dot(G1, np.dot(G2, G3))

summ = 0
# So a1 and a2 are the auxilary indices! 
for a1 in range(7):
    for a2 in range(3):
        summ += G1[0,a1]*G2[a1,a2]*G3[a2,0]
        
print("If %d is the same as %d, then I figured it out!!" % (summ, hej))
