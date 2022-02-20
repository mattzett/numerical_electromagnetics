#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 07:29:12 2020

Numerical soluation to a 2D potential problem similar to those from class

@author: zettergm
"""

import numpy as np
from ittools import Jacobi,GaussSeidel
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg

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
M=np.zeros( (N,N) )    # solutions are miserably slow using sparse storage for some reason...
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
[PhiJ,iteration]=Jacobi(Phi0,M,rhs,tol,False)
print("---------------------------------------------------------------------")
print("Jacobi Iterative solution done...")
print(x)
print("Number of iterations required and tolerance")
print(iteration)
print(tol)

# Gauss-Seidel iterative solution, usually will use fewer iterations than Jacobi
print("---------------------------------------------------------------------")
[PhiGS,iteration]=GaussSeidel(Phi0,M,rhs,tol,False)
print("---------------------------------------------------------------------")
print("Gauss-Seidel Iterative solution done...")
print(x)
print("Number of iterations required and tolerance")
print(iteration)
print(tol)

# Solution with umfpack, note how fast this is compared to the iterative solutions :)
print("---------------------------------------------------------------------")
Msparse=scipy.sparse.csr_matrix(M)    
# normally more efficient to make the csr matrix on a per-entry basis 
#   but we alread have the full version....
rhssparse=scipy.sparse.csr_matrix(np.reshape(rhs,[N,1]))
PhiUMF=scipy.sparse.linalg.spsolve(Msparse,rhssparse,use_umfpack=True)
print("---------------------------------------------------------------------")
print("Solution with UMFPACK done...")

# reorganize solution data
PhiJ=np.reshape(PhiJ, (lx,ly))
PhiGS=np.reshape(PhiGS, (lx,ly))
PhiUMF=np.reshape(PhiUMF, (lx,ly))

# plot
plt.subplots(1,3,figsize=(14,6),dpi=100)
plt.subplot(1,3,1)
plt.pcolormesh(x,y,PhiJ,shading="auto")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Jacobi [V]")
plt.colorbar()

plt.subplot(1,3,2)
plt.pcolormesh(x,y,PhiGS,shading="auto")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gauss-Seidel [V]")
plt.colorbar()

plt.subplot(1,3,3)
plt.pcolormesh(x,y,PhiUMF,shading="auto")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sparse LU factorization [V]")
plt.colorbar()



