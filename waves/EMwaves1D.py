#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:02:17 2022

Solution for electric and magnetic fields of a plane wave along the axis of propagation

@author: zettergm
"""

# imports
import numpy as np
from numpy import exp,sqrt,pi,sin
import matplotlib.pyplot as plt
from hyptools import vecLaxWen

# material parameters
mu=4*pi*1e-7
eps=8.854e-12
v=1/sqrt(mu*eps)    # speed of light in this medium
Z=sqrt(mu/eps)      # wave impedance in this medium

# propagation matrix for fields
A=np.zeros( (2,2) )
A[0,1]=1/eps
A[1,0]=1/mu

# Define a 1D space and time grid in x,t for a test problem
lz=96
a=0     # here a,b are the endpoints of the z-domain; we'll assume periodic boundary conditions to simplify solutions
b=1
z=np.linspace(a,b,lz);
dz=z[1]-z[0];        #grid spacing
targetCFL=0.5        # how close to run to the limits of marginal stability for explicit techhniques
dt=targetCFL*dz/v
N=192                 # number of time steps to take
t=np.arange(0,N*dt,dt)
lt=t.size

# Wave packet sizes
sigma=(b-a)/7.5    # gaussian group envelope
zavg=1/2*(a+b)
k=2*pi/sigma      # wavenumber

# initial conditions
Ex=exp(-(z-zavg)**2/2/sigma**2)*sin(k*z)
Ex=np.reshape(Ex, (1,lz) )
Hy=Ex/Z
Hy=np.reshape(Hy, (1,lz) )
fields=np.concatenate( (Ex,Hy), axis=0)    # create field matrix

# iterate over time to solve PDE and plto
maxE=np.max(Ex)
maxH=np.max(Hy)
for n in range(0,lt):
    fields=vecLaxWen(dt,dz,A,fields)
    Ex=fields[0,:]
    Hy=fields[1,:]
    
    # plot results of each time step and pause briefly
    plt.subplots(2,1,num=1,dpi=150)
    plt.clf()
    #
    plt.subplot(2,1,1)
    plt.plot(z,Ex)
    plt.ylabel("$E_x$")
    plt.title( "$t$ = %e s" % ( (n+1)*dt) )
    plt.ylim((-maxE,maxE))
    plt.xlim((a,b))
    #
    plt.subplot(2,1,2)
    plt.plot(z,Hy)
    plt.xlabel("$z$")
    plt.ylabel("$H_y$")
    plt.ylim((-maxH,maxH))
    plt.xlim((a,b))    
    #
    plt.show()
    if n==0:
        plt.pause(0.5)
    else:
        plt.pause(0.01)
    