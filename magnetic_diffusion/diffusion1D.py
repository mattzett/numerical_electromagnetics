#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:43:50 2022

Illustrating a basic transient magnetic diffusion problem

@author: zettergm
"""

import numpy as np
import scipy.sparse.linalg
import scipy.sparse
import matplotlib.pyplot as plt
import time

# Material parameters
mu=1
sigma=1

# Size of grid
lx=100
Nmax=100
x=np.linspace(-10,10,lx)
dx=x[1]-x[0]
dt = 0.05

# Matrix defining finite-difference equation for laplacian operator
#   This could definitely benefit for sparse storage and a banded/tridiagonal solver
#A=np.exp(-(x**2/2))
A=np.zeros(lx)
A[lx//2-lx//6:lx//2+lx//6]=1
M=np.zeros( (lx,lx) )
rhs=np.zeros(lx)
for n in range(0,Nmax):
    for i in range(0,lx):         
        if i==0:
            M[i,i]=1
            rhs[i]=0
        elif i==lx-1:
            M[i,i]=1
            rhs[i]=0
        else:
            M[i,i-1]=-dt/dx**2
            M[i,i]=1+2*dt/dx**2
            M[i,i+1]=-dt/dx**2
            rhs[i]=A[i]
    Msparse=scipy.sparse.csr_matrix(M)    
    rhssparse=scipy.sparse.csr_matrix(np.reshape(rhs,[lx,1]))
    A=scipy.sparse.linalg.spsolve(Msparse,rhssparse,use_umfpack=True)
    
    plt.figure(1,dpi=150)
    plt.clf()
    plt.plot(x,A)
    plt.xlabel("x")
    plt.ylabel("A(x)")
    plt.ylim((0,1))
    plt.show()
    plt.pause(0.01)
    