#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:07:23 2020

@author: Tobias

Just playing around getting to know tensorly a little better.
"""

import tensorly as tl
from tensorly.decomposition import parafac, tucker, non_negative_parafac
import numpy as np
import matplotlib.pyplot as plt

# Initial tensor is a "box" with every number equal to the value of the 3 dimensions

A = tl.tensor(
    (((111,121),
      (211,221),
      (311,321),
      (411,321)),
     ((112,122),
      (212,222),
      (312,322),
      (412,422)),
     ((113,123),
      (213,223),
      (313,323),
      (413,423))))
print(A)
print("The dimensions are: ",A.shape)

#%% Unfolding
print("Mode 1: \n",tl.unfold(A,1))
print("Mode 2: \n",tl.unfold(A,2))
print("Mode 3: \n",tl.unfold(A,0))

"""
Mode 1:
    Corresponds to concatenating the frontal matrices
Mode 2:
    Corresponds to concatenating the frontal matrices transposed
Mode 3:
    Corresponds to making each frontal matrices into a row vector, by 
    concatenating the rows of the matrix.
"""

#%% Folding is the opposite
print(tl.fold(tl.unfold(A,2),2,A.shape))

# %% Tensor decompositioning
from mpl_toolkits.mplot3d import Axes3D
""" 
I will try to decompose a 3-dimensional function to see if this algorithm can 
capture the individual tendencies
"""

def fun(x,y):
    return 0.5 * y**2 * np.sin(x) + 4*x

xs = np.arange(0,10,0.1)
ys = np.arange(0,10,0.1)
X, Y = np.meshgrid(xs,ys)
Z = fun(X,Y)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)

# %% Trying to decompose Z

from numpy.linalg import pinv, inv

def frobenius(X):
    return(np.sqrt(np.sum(X**2)))

def corcondia(t, G):
    return 100 * (1 - ((t - G)**2).sum() / (t**2).sum())

dataTensor = tl.tensor(Z)
numComp = 2
#kruskalTensor = parafac(dataTensor,rank=numComp)
kruskalTensor = tucker(dataTensor, ranks = [2,2])
loadings = kruskalTensor[1]

A = loadings[0]
B = loadings[1]

plt.figure(2)
plt.subplot(1,2,1)
plt.plot(ys,B)
plt.subplot(1,2,2)
plt.plot(xs,A)

#%%
#Z_hat = tl.kruskal_to_tensor(kruskalTensor)
Z_hat = tl.tucker_to_tensor(kruskalTensor)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z-Z_hat)
print(frobenius(Z-Z_hat))

# %% Assessing the performance of the decomposition
from tensorly.tenalg import multi_mode_dot
G = multi_mode_dot(Z, [pinv(A), pinv(B)], modes = [0,1])

print(corcondia(np.eye(numComp), G))