#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:19:03 2022

Illustrate various special functions using scipy

@author: zettergm
"""

# imports
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import scipy.special as scspec

# clear figures
plt.close("all")


## Legendre polynomials, compute and store
lmax=7    # compute legendre polynomials up to lmax-1
lx=100    # number of datapoints on the x-axis
x=np.linspace(-1,1,lx)    # full range for cos theta
Pell=np.zeros( (lmax+1,lx) )
for ell in range(0,lmax):
    pobj=scspec.legendre(ell)
    Pell[ell,:]=np.polyval(pobj,x)

# Plot the Legendre polynomials
plt.figure(dpi=150)
legnd=[]
for ell in range(0,lmax):
    plt.plot(x,Pell[ell,:])
    legnd.append("$\\ell=$"+str(ell))
plt.xlabel("x")
plt.ylabel("$P_\\ell (x)$")
plt.legend(legnd)
plt.title("Legendre Polynomials")


## Bessel functions
mmax=7
lkp=100
kp=x=np.linspace(0,10,lkp)
Jm=np.zeros( (mmax,lkp) )
for m in range(0,mmax):
    Jm[m,:]=scspec.jv(m,kp)

# plot the Bessel functions
plt.figure(dpi=150)
legnd=[]
for m in range(0,mmax):
    plt.plot(kp,Jm[m,:])
    legnd.append("$m=$"+str(m))
plt.xlabel("$k \\rho$")
plt.ylabel("$J_m (k \\rho)$")
plt.legend(legnd)
plt.title("Bessel functions")


## Spherical Bessel functions
lmax=7
lkr=100
kr=x=np.linspace(0,10,lkr)
jell=np.zeros( (lmax,lkr) )
for ell in range(0,lmax):
    jell[ell,:]=scspec.spherical_jn(ell,kr)

# plot the spherical Bessel functions
plt.figure(dpi=150)
legnd=[]
for ell in range(0,lmax):
    plt.plot(kr,jell[ell,:])
    legnd.append("$\\ell=$"+str(ell))
plt.xlabel("$kr$")
plt.ylabel("$j_\\ell (k r)$")
plt.legend(legnd)
plt.title("Spherical Bessel functions")


## Spherical harmonics
lmax=6
ltheta=100
lphi=200
theta=np.linspace(0,pi,ltheta)
phi=np.linspace(0,2*pi,lphi)
[PHI,THETA]=np.meshgrid(phi,theta)
Yellm=np.zeros( (ltheta,lphi) )    # don't try to store all of these, just plot as we go
for ell in range(0,lmax):
    for m in range(0,ell+1):
        Yellm=scspec.sph_harm(m,ell,PHI,THETA)
        plt.subplots(dpi=150)
        plt.subplot(1,2,1)
        plt.pcolormesh(phi,theta,np.real(Yellm),shading="auto")
        plt.xlabel("$\\phi$")
        plt.ylabel("$\\theta$")
        plt.title("$\\Re Y_{\\ell m} (\\theta,\\phi); \\ell=$"+str(ell)+",$ m=$"+str(m))
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.pcolormesh(phi,theta,np.imag(Yellm),shading="auto")
        plt.xlabel("$\\phi$")
        plt.title("$\\Im Y_{\\ell m} (\\theta,\\phi); \\ell=$"+str(ell)+",$ m=$"+str(m))        
        plt.colorbar()
        
        