#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:07:40 2022

Tools for doing time stepping for parabolic equations

@author: zettergm
"""

# imports
import numpy as np
import scipy.sparse

# function to setup a (constant) matrix kernel for soluving 
def matrix_kernel(lz,dt,dz,D):
    # Matrix defining finite-difference equation for laplacian operator, one-time setup for this problem
    M=np.zeros( (lz,lz) )
    for i in range(0,lz):         
        if i==0:
            M[i,i]=1
        elif i==lz-1:
            M[i,i]=1
        else:
            M[i,i-1]=-D*dt/dz**2
            M[i,i]=1+2*D*dt/dz**2
            M[i,i+1]=-D*dt/dz**2
    Msparse=scipy.sparse.csr_matrix(M)
    return Msparse


