#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 07:29:12 2020

2D potential problem similar to those from class

@author: zettergm
"""

import numpy as np
from ittools import Jacobi,GaussSeidel
import matplotlib.pyplot as plt
import scipy.sparse

# parameters of problem:
lx=24; ly=24;
N=lx*ly    # total number of grid points
a=1; b=1;   # square region

# create a 2D grid
x=np.linspace(0,a,lx)
y=np.linspace(0,b,ly)
dx=x[1]-x[0]
dy=y[1]-y[0]
[X,Y]=np.meshgrid(x,y)

# Dirichlet boundary condition for four sides of square
f1=np.ones( (lx) )
f2=np.ones( (lx) )
g1=np.zeros( (ly) )
g2=np.zeros( (ly) )

# Right-hand side of Poisson equation, viz. rho/eps0
rhs=np.zeros( (N) )

# Matrix defining finite-difference equation for laplacian operator
M=np.zeros( (N,N) )
for i in range(0,lx):
    for j in range(0,ly):
        k=j*lx+i    # linear index referencing i,j grid point
        if j==0:
            M[k,k]=1
            rhs[k]=f1[i]
        elif j==ly-1:
            M[k,k]=1
            rhs[k]=f2[i]            
        elif i==0:
            M[k,k]=1
            rhs[k]=g1[j]
        elif i==lx-1:
            M[k,k]=1
            rhs[k]=g2[j]
        else:
            M[k,k-lx]=1/dy**2
            M[k,k-1]=1/dx**2
            M[k,k]=-2/dx**2-2/dy**2
            M[k,k+1]=1/dx**2
            M[k,k+lx]=1/dy**2
            rhs[k]=0


# Jacobi iterative solution
Phi0=np.ones( (N) )    # use a bunch of ones for the inital guess
tol=1
print("---------------------------------------------------------------------")
[Phi,iteration]=Jacobi(Phi0,M,rhs,tol,False)
print("---------------------------------------------------------------------")
print("Jacobi Iterative solution")
print(x)
print("Number of iterations required and tolerance")
print(iteration)
print(tol)

# Gauss-Seidel iterative solution
print("---------------------------------------------------------------------")
[Phi,iteration]=GaussSeidel(Phi0,M,rhs,tol,False)
print("---------------------------------------------------------------------")
print("Gauss-Seidel Iterative solution")
print(x)
print("Number of iterations required and tolerance")
print(iteration)
print(tol)

# reorganize solution data and plot
Phi=np.reshape(Phi, (lx,ly))
plt.figure(dpi=150)
plt.pcolormesh(x,y,Phi,shading="auto")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Numerical solution for 2D potential problem")
plt.colorbar()

# print("Built-in python solution")
# xpyth=np.linalg.solve(A,b)
# print(xpyth)
# print("Residual:  ")
# print(x-xpyth)