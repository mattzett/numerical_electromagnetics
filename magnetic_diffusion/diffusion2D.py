#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:02:18 2022

Numerical solution to a 2D diffusion problem.  Note that this is a problem in 
the course that is solved in spherical coordinates, but in this case we are
doing the numerical solution in Cartesian.  This code models the diffusion
of magnetic field from an initially uniformly magnetized sphere thorugh a 
conducting surrounding regions.  

@author: zettergm
"""

# imports
import numpy as np
from numpy import pi,exp
from difftools import matrix_kernel
import scipy.sparse.linalg
import matplotlib.pyplot as plt

# Material parameters
mu=4*pi*1e-7
sigma=1e6
D=1/mu/sigma    # equivalent diffusion coefficient
a=1
H0=1
B0=1
nu=1/mu/sigma/a**2
moment=2*pi*a**2*B0/mu

# by design we have a constant delta for both dimensions, same number of points
#   in each direction and same grid step size
lz=100
Nmax=50
z=np.linspace(-4*a,4*a,lz)
x=z
[X,Z]=np.meshgrid(x,z)
R=np.sqrt(X**2+Z**2)
THETA=np.arccos(Z/R)
dz=z[1]-z[0]
dx=dz
dt = 5*dz**2/D/2    # explicit stabilty limit will results in really slow time stepping; use 5 times larger.  

# We use a single matrix kernel to do the time-stepping.  This is possible due to the
#   same number of grid points in each diretion and same grid step size.
Msparse=matrix_kernel(lz,dt,dz,D)

# Initial conditions
#Hy=H0*exp(-X**2/2/(a/2)**2)*exp(-Z**2/2/(a/2)**2)
Hy=np.zeros( (lz,lz) )
for i in range(0,lz):
    for j in range(0,lz):
        if R[i,j]<a:    # internal multipole moment calculation
            Hy[i,j]=mu/3*(1-1/2/mu)*B0*R[i,j]*(-np.sin(THETA[i,j]))
        else:    # outside the sphere it should be a dipole
            Hy[i,j]=mu*moment/4/pi/R[i,j]**2*np.sin(THETA[i,j])
Hmax=np.max(Hy)

# time iterations
for n in range(0,Nmax):
    # do the solves for the x-direction first
    for i in range(0,lz):
        Hyslice=Hy[i,:]
        rhsslice=Hyslice
        rhsslice[0]=0    # fix the ends
        rhsslice[-1]=0
        rhssparse=scipy.sparse.csr_matrix(np.reshape(rhsslice,[lz,1]))
        Hyslice=scipy.sparse.linalg.spsolve(Msparse,rhssparse,use_umfpack=True)   # umfpack is overkill for this but will presumably work
        Hy[i,:]=Hyslice
    for i in range(0,lz):
        Hyslice=Hy[:,i]
        rhsslice=Hyslice
        rhsslice[0]=0    # fix the ends
        rhsslice[-1]=0
        rhssparse=scipy.sparse.csr_matrix(np.reshape(rhsslice,[lz,1]))
        Hyslice=scipy.sparse.linalg.spsolve(Msparse,rhssparse,use_umfpack=True)   # umfpack is overkill for this but will presumably work
        Hy[:,i]=Hyslice      
        
    # plot results of each time step and pause briefly
    plt.subplots(1,3,num=1,dpi=100,figsize=(14,5))
    plt.clf()
    #
    plt.subplot(1,3,1)
    plt.pcolormesh(x,z,Hy,shading="auto")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.ylabel("$A_y(x,z)$")
    plt.title( "$t$ = %6.4f s" % ( (n+1)*dt) )
    plt.ylim((-2*a,2*a))
    plt.xlim((-2*a,2*a))
    plt.colorbar(orientation="horizontal")
    plt.clim(0,0.667*Hmax)
    #
    # add a line for the sphere surface
    phi=np.linspace(0,2*pi,100)
    plt.plot(a*np.cos(phi),a*np.sin(phi),"w")
    #
    plt.subplot(1,3,2)
    plt.plot(z,Hy[:,lz//2])
    plt.xlim((-2*a,2*a))
    plt.xlabel("$z$")
    plt.ylabel("$H_y$")
    plt.ylim((0,Hmax))
    #
    plt.subplot(1,3,3)
    plt.plot(x,Hy[lz//2,:])
    plt.xlim((-2*a,2*a))
    plt.xlabel("$x$")
    plt.ylabel("$H_y$")
    plt.ylim((0,Hmax))    
    #
    plt.show()
    if n==0:
        plt.pause(0.5)
    else:
        plt.pause(0.025)
    
    
    
    