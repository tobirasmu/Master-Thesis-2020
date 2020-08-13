#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:07:23 2020

@author: Tobias

Just playing around getting to know tensorly a little better.
"""

import tensorly as tl
import numpy as np

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


