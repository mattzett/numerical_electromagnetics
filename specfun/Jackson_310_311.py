#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 18:23:48 2022

This code computes and plots the series solutions to Problems 3.10-3.11 of Jackson

@author: zettergm
"""

# imports
import numpy as np
from numpy import pi,sin,cos
import matplotlib.pyplot as plt
import scipy.special as scspec

# clear figures
plt.close("all")

# geometry of cylinder for problems 3.10-3.11
b=1; L=1;

# potential boundary amplitude
V=1

# size of computational domain, for plotting we want to visualize in Cartesian space
lx=100; ly=100; lz=50;
Phi=np.zeros( (lx,ly,lz) )
x=np.linspace(-b,b,lx)
y=np.linspace(-b,b,ly)
z=np.linspace(0,L,lz)
[X,Y,Z]=np.meshgrid(x,y,z)

# for calculations our series solution is in terms of cylindrical variables
rho=np.sqrt(X**2+Y**2)
phi=np.arctan2(Y,X)

# for plotting purposes we want to mask the region outside the cylinder
indsout=rho>b   # logical array having true if point is outside domain of interest

# compute coefficients
mmax=21
nmax=21
for m in range(1,mmax+1,2):
    for n in range(1,nmax+1,2):
        print("Computing m,n values:  ",m,n)
        kn=n*pi/L
        Imb=scspec.iv(m,kn*b);
        Anm=(-1)**((m-1)/2)*16*V/m/n/pi**2/Imb
        
        Im=scspec.iv(m,kn*rho)
        Phi=Phi+Anm*sin(kn*Z)*cos(m*phi)*Im
    
# get rid of regions where solution isn't valid
Phi[indsout]=np.nan

# plot the results along some sensible axes, just choose middle for now
iz=lz//2
iy=ly//2
plt.subplots(dpi=150)
plt.subplot(1,2,1)
plt.pcolormesh(x,y,Phi[:,:,iz],shading="auto")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("$\Phi$ [V]")
plt.subplot(1,2,2)
plt.pcolormesh(x,z,Phi[iy,:,:].transpose(),shading="auto")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("z")
plt.title("$\Phi$ [V]")