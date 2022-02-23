#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:14:11 2022

Run a time dependent wave simulation

@author: zettergm
"""

# imports
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from hyptools import LaxWen

# Define a 1D space and time grid in x,t for a test problem
lz=64
a=0     # here a,b are the endpoints of the x-domain
b=1
z=np.linspace(a,b,lz);
dz=z[1]-z[0];        #grid spacing
v=1                  # velocity of wave propagation
targetCFL=0.9        # how close to run to the limits of marginal stability for explicit techhniques
dt=targetCFL*dz/v
N=75                 # number of time steps to take
t=np.arange(0,N*dt,dt)
lt=t.size
sigma=(b-a)/10
zavg=1/2*(a+b)

Ex=exp(-(z-zavg)**2/2/sigma**2)
maxE=np.max(Ex)
for n in range(0,lt):
    Ex=LaxWen(dt,dz,v,Ex)
    
    # plot results of each time step and pause briefly
    plt.figure(1,dpi=150)
    plt.clf()
    plt.plot(z,Ex)
    plt.xlabel("$x$")
    plt.ylabel("$E_x$")
    plt.title( "$t$ = %6.4f s" % ( (n+1)*dt) )
    plt.ylim((0,maxE))
    plt.xlim((a,b))
    plt.show()
    plt.pause(0.01)