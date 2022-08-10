#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:43:50 2022

Illustrating a basic transient magnetic diffusion problem, See Jackson Section 5.18

@author: zettergm
"""

import numpy as np
import scipy.sparse.linalg
import scipy.sparse
from scipy.special import erf
import matplotlib.pyplot as plt
from numpy import pi,sqrt,abs
from difftools import matrix_kernel

# Material parameters
mu=4*pi*1e-7
sigma=1e6
D=1/mu/sigma    # equivalent diffusion coefficient
a=1
H0=1
nu=1/mu/sigma/a**2

# Size of grid
lz=250
Nmax=200
z=np.linspace(-5*a,5*a,lz)
dz=z[1]-z[0]
dt = 5*dz**2/D/2    # explicit stabilty limit will results in really slow time stepping; use 5 times larger.  

#   This could definitely benefit for sparse storage and a banded/tridiagonal solver
#A=np.exp(-(x**2/2))
Hx=np.zeros(lz)
indmin=np.argmin(abs(z+a))
indmax=np.argmin(abs(z-a))
Hx[indmin:indmax]=1

# Matrix defining finite-difference equation for laplacian operator, one-time setup for this problem
Msparse=matrix_kernel(lz,dt,dz,D)
rhs=np.zeros( (lz,1) )

# time iterations
for n in range(0,Nmax):
    # set up time-dependent part of the problem and solve
    for i in range(1,lz-1):
        rhs[i]=Hx[i]
    rhssparse=scipy.sparse.csr_matrix(np.reshape(rhs,[lz,1]))
    Hx=scipy.sparse.linalg.spsolve(Msparse,rhssparse,use_umfpack=True)   # umfpack is overkill for this but will presumably work

    # Solution from Jackson eqn. 5.176
    HxJ=H0/2*( erf((1+abs(z)/a)/2/sqrt((n+1)*dt*nu)) + erf((1-abs(z)/a)/2/sqrt((n+1)*dt*nu)) )

    # plot results of each time step and pause briefly
    plt.figure(1,dpi=150)
    plt.clf()
    plt.plot(z,HxJ,'o')
    plt.plot(z,Hx)
    plt.xlabel("$x$")
    plt.ylabel("$H_x(z)$")
    plt.title( "$t$ = %6.4f s" % ( (n+1)*dt) )
    plt.ylim((0,H0))
    plt.xlim((-2*a,2*a))
    plt.legend( ("Jackson 5.176","Numerical BTCS") )
    plt.show()
    if (n==0):
        plt.pause(0.5)
    else:
        plt.pause(0.01)

