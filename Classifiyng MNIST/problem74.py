#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:53:10 2020

@author: Tobias
"""
import time as T

def factorial(x):
    if (x == 0):
        return(1)
    return(x*factorial(x-1))

factorials = []
for i in range(10):
    factorials.append(factorial(i))

def makeChain(n):
    chain = [n]
    while (True):
        this = str(chain[-1])
        new = sum([factorials[int(x)] for x in this])
        if (new in chain):
            chain = (len(chain),chain,new)
            break
        chain.append(new)
    return(chain)

# Calculating for every starting number below 1.000.000
start = T.time()
answer = 0
for i in range(100,1000000):
    if (i % 10000 == 0):
        print("Iterations: ",i)
    if (makeChain(i)[0] == 60):
        answer += 1
slut = T.time()
print("There are: ",answer, " starting numbers resulting in 60 non-repeating terms. Found in: ",round(slut-start,1)," seconds")