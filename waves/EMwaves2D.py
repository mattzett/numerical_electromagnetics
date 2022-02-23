#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:38:35 2022

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
Z0=sqrt(mu/eps)      # wave impedance in this medium

# propagation matrices for fields in each direction
Az=np.zeros( (3,3) )
Az[0,2]=1/eps
Az[2,0]=1/mu
Ax=np.zeros( (3,3) )
Ax[1,2]=-1/eps
Ax[2,1]=-1/mu

# Define a 1D space and time grid in x,t for a test problem
lz=96
a=0     # here a,b are the endpoints of the z-domain; we'll assume periodic boundary conditions to simplify solutions
b=1
z=np.linspace(a,b,lz)
dz=z[1]-z[0];        #grid spacing
x=np.linspace(a,b,lz)
dx=x[1]-x[0]
[X,Z]=np.meshgrid(x,z)

# controlling time iterations
targetCFL=0.5        # how close to run to the limits of marginal stability for explicit techhniques
dt=targetCFL*dz/v
N=32                # number of time steps to take
t=np.arange(0,N*dt,dt)
lt=t.size

# Wave packet sizes
sigma=(b-a)/7.5    # gaussian group envelope
zavg=1/2*(a+b)
xavg=1/2*(a+b)
k=2*pi/sigma      # wavenumber

# initial conditions
Ex=exp(-(Z-zavg)**2/2/sigma**2)*exp(-(X-xavg)**2/2/sigma**2)
Ez=-exp(-(Z-zavg)**2/2/sigma**2)*exp(-(X-xavg)**2/2/sigma**2)
Ex=np.reshape(Ex, (1,lz,lz) )
Ez=np.reshape(Ez, (1,lz,lz) )
Hy=-Ez/Z0/sqrt(2)+Ex/Z0/sqrt(2)
fields=np.concatenate( (Ex,Ez,Hy), axis=0)    # create field matrix

# iterate over time to solve PDE and plto
Emax=np.max(Ex)
Hmax=np.max(Hy)
for n in range(0,lt):
    # sweep the x-direction
    for i in range(0,lz):
        fields[:,i,:]=vecLaxWen(dt,dx,Ax,fields[:,i,:])
    # sweep the z-direction
    for j in range(0,lz):
        fields[:,:,j]=vecLaxWen(dt,dz,Az,fields[:,:,j])
    
    # "copy" out the fields into a sensibly named variable
    Ex=fields[0,:,:]
    Ez=fields[1,:,:]
    Hy=fields[2,:,:]
    
    # plot results of each time step and pause briefly
    plt.subplots(1,3,num=1,dpi=100,figsize=(14,5))
    plt.clf()
    #
    plt.subplot(1,3,1)
    plt.pcolormesh(x,z,Ex,shading="auto")
    plt.xlabel("$x$")
    plt.ylabel("$z$")
    plt.title( "$E_x; t$ = %e s" % ( (n+1)*dt) )
    plt.colorbar()
    plt.clim(-Emax,Emax)
    #
    plt.subplot(1,3,2)
    plt.pcolormesh(x,z,sqrt(Ez**2+Ex**2),shading="auto")
    plt.xlabel("$x$")
    plt.ylabel("$z$")
    plt.title("$E_z$")
    plt.colorbar()
    plt.clim(-Emax,Emax)    
    #
    plt.subplot(1,3,3)
    plt.pcolormesh(x,z,np.abs(Hy),shading="auto")
    plt.xlabel("$x$")
    plt.ylabel("$z$")
    plt.title("$H_y$")
    plt.colorbar()
    plt.clim(-Hmax,Hmax)
    #
    plt.show()
    plt.pause(0.01)