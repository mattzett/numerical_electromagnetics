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
from numpy import pi

# Material parameters
mu=4*pi*1e-7
sigma=1e6
D=1/mu/sigma    # equivalent material coefficient
a=1

# Size of grid
lx=250
Nmax=100
x=np.linspace(-5*a,5*a,lx)
dx=x[1]-x[0]
dt = 5*dx**2/D/2

# Matrix defining finite-difference equation for laplacian operator
#   This could definitely benefit for sparse storage and a banded/tridiagonal solver
#A=np.exp(-(x**2/2))
A=np.zeros(lx)
indmin=np.argmin(abs(x+a/2))
indmax=np.argmin(abs(x-a/2))
A[indmin:indmax]=1
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
            M[i,i-1]=-D*dt/dx**2
            M[i,i]=1+2*D*dt/dx**2
            M[i,i+1]=-D*dt/dx**2
            rhs[i]=A[i]
    Msparse=scipy.sparse.csr_matrix(M)    
    rhssparse=scipy.sparse.csr_matrix(np.reshape(rhs,[lx,1]))
    A=scipy.sparse.linalg.spsolve(Msparse,rhssparse,use_umfpack=True)   # umfpack is overkill for this but will presumably work
    
    plt.figure(1,dpi=150)
    plt.clf()
    plt.plot(x,A)
    plt.xlabel("x")
    plt.ylabel("A(x)")
    plt.title( "t = %6.4f s" % (n*dt) )
    plt.ylim((0,1))
    plt.xlim((-2*a,2*a))
    plt.show()
    plt.pause(0.01)
    