#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 20:18:24 2025

@author: zettergm
"""

import numpy as np
from numpy import pi
from elliptical_coords import elliptical2cart,cart2elliptical
import matplotlib.pyplot as plt

lmu=64
lnu=256

a=1
e=10
c=e*a
b=np.sqrt(a**2+c**2)

mu=np.linspace(0,0.25,lmu)
nu=np.linspace(0,2*pi,lnu)

MU,NU=np.meshgrid(mu,nu,indexing='ij')

X,Y=elliptical2cart(MU,NU,a,b)

# Visualize whether this seems to be a good description of elliptical coords
plt.figure()
plt.plot(X,Y,'.')
plt.show()

# Test whether transformation is unitary
MUtest,NUtest = cart2elliptical(X,Y,a,b)
mutest=MUtest[:,1]
nutest=NUtest[1,:]

# Visualize transformed coords.
plt.figure()
plt.plot(mu-mutest)
plt.plot(nu-nutest)
plt.show()
