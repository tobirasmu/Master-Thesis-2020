#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:50:28 2020

@author: Tobias
"""
import numpy as np

# Problem 75 of project euler 

Ls = np.zeros(1500001)

for a in range(1,750000):
    print(a)
    for b in range(a+1,750000):
        c = np.sqrt(a**2 + b**2)
        S = a + b + int(c)
        if (round(c) == c and S <= 1500000):
            Ls[S] += 1
        if (S > 1500000):
            break

#%%
import numpy as np     
from functools import reduce

def factor(n):
    factors = []
    nSq = int(n**0.5)
    nTest = nSq*(2**0.5)
    for i in range(nSq,0,-1):
        if (n % i == 0):
            a = int(n/i)
            b = int(i)
            factors.append(a)
            factors.append(b)
            if (3*nTest+2*(a+b) > 1500000):
                break
    return factors

def dickson(r):
    triples = []
    factors = factor(int((r**2)/2))
    for i in range(0,len(factors),2):
        s = factors[i]
        t = factors[i+1]
        triples.append([r+s,r+t,r+s+t])
    return(triples)
    
Ls = np.zeros(1500001)
for r in range(2,500001,2):
    if (r% 1000 == 0):
        print("Iteration: ",r)
    triples = dickson(r)
    for i in range(0,len(triples)):
        this = sum(triples[i])
        if (this > 1500000):
            break
        else:
            Ls[this] += 1

