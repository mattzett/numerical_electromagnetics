#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 20:31:56 2025

@author: zettergm
"""

import numpy as np
#from ittools import Jacobi,GaussSeidel
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg


# parameters of problem:
lx=128; ly=128;
N=lx*ly             # total number of grid points
a=4; b=4;           # x,y extents
Ey0=-0.1            # background field in which the object is immersed
Ex0=0.0

# create a 2D grid
x=np.linspace(-a,a,lx)
y=np.linspace(-b,b,ly)
dx=x[1]-x[0]
dy=y[1]-y[0]
[X,Y]=np.meshgrid(x,y,indexing='ij')

# cylindrical coordinates
semimajor=2      # >1 makes semimajor axis in y-direction
rho=np.sqrt(X**2+Y**2/semimajor**2)
phi=np.arctan2(Y,X)

# Dirichlet boundary condition for four sides of square
#f1=np.zeros( (lx) )
#f2=np.ones( (lx) )
f1=np.zeros( (lx) )
f2=np.zeros( (lx) )
#g1=np.zeros( (ly) )
#g2=np.zeros( (ly) )
potmax=-Ey0*(y[-1]-y[0])
#g1=np.linspace(0,1,ly)
#g2=np.linspace(0,1,ly)
g1=np.linspace(0,potmax,ly)
g2=g1

# Density structure(s) on grid
n0=4e11
n1=1e11
rho0=0.3
L=0.2
rho1=rho0-L*np.log(n1/n0)    # solution for end of gradient region given a starting point and scale length
n = np.zeros( (lx,ly) )
for i in range(0,lx):
    for j in range (0,ly):
        if rho[i,j] < rho0:
            n[i,j]=n0
        elif rho[i,j] > rho0 and rho[i,j] < rho1: 
            n[i,j]=n0*np.exp(-(rho[i,j]-rho0)/L)    # ODE solution for density of fixed scale length
        else:
            n[i,j]=n1

# Ballpark the number of cells in the gradient region
gradcells=int(np.floor((rho1-rho0)/dy))
print("Rough width (cells) of transition region:  ",gradcells)

# Density gradients
[dndx,dndy]=np.gradient(n,x,y)

# Right-hand side of Poisson equation, viz. -rho/eps0
rhs=np.zeros( (N) )

# Matrix defining finite-difference equation for laplacian operator
M=np.zeros( (N,N) )    # solutions are miserably slow using sparse storage for some reason...
for i in range(0,lx):
    for j in range(0,ly):
        k=j*lx+i    # linear index referencing i,j grid point
        if j==0:
            M[k,k]=-1/dy
            M[k,k+lx]=1/dy
            rhs[k]=-Ey0
        elif j==ly-1:
            M[k,k-lx]=-1/dy
            M[k,k]=1/dy
            rhs[k]=-Ey0           
        elif i==0:
            M[k,k]=1
            rhs[k]=g1[j]
        elif i==lx-1:
            M[k,k]=1
            rhs[k]=g2[j]
        else:
            M[k,k-lx]=1/dy**2 - 1/(2*dy)*dndy[i,j]/n[i,j]         # i,j-1
            M[k,k-1]=1/dx**2  - 1/(2*dx)*dndx[i,j]/n[i,j]         # i-1,j
            M[k,k]=-2/dx**2-2/dy**2                               # i,j
            M[k,k+1]=1/dx**2 + 1/(2*dx)*dndx[i,j]/n[i,j]          # i+1,j
            M[k,k+lx]=1/dy**2 + 1/(2*dy)*dndy[i,j]/n[i,j]         # i,j+1
            rhs[k]=0


# # Jacobi iterative solution
# Phi0=np.ones( (N) )    # use a bunch of ones for the inital guess
# tol=1
# print("---------------------------------------------------------------------")
# [PhiJ,iteration]=Jacobi(Phi0,M,rhs,tol,False)
# print("---------------------------------------------------------------------")
# print("Jacobi Iterative solution done...")
# print(x)
# print("Number of iterations required and tolerance")
# print(iteration)
# print(tol)

# # Gauss-Seidel iterative solution, usually will use fewer iterations than Jacobi
# print("---------------------------------------------------------------------")
# [PhiGS,iteration]=GaussSeidel(Phi0,M,rhs,tol,False)
# print("---------------------------------------------------------------------")
# print("Gauss-Seidel Iterative solution done...")
# print(x)
# print("Number of iterations required and tolerance")
# print(iteration)
# print(tol)

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
#PhiJ=np.reshape(PhiJ, (lx,ly))
#PhiGS=np.reshape(PhiGS, (lx,ly))
PhiUMF=np.reshape(PhiUMF, (lx,ly), order='F')
[ExUMF,EyUMF]=np.gradient(-PhiUMF,x,y)

# plot
# plt.subplots(1,3,figsize=(14,6),dpi=100)
# plt.subplot(1,3,1)
# plt.pcolormesh(x,y,PhiJ,shading="auto")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Jacobi [V]")
# plt.colorbar()

# plt.subplot(1,3,2)
# plt.pcolormesh(x,y,PhiGS,shading="auto")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Gauss-Seidel [V]")
# plt.colorbar()

plt.subplots(1,3,figsize=(14,6),dpi=100)
plt.subplot(1,3,1)
plt.pcolormesh(x,y,PhiUMF.transpose(),shading="auto")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sparse LU factorization [V]")
plt.colorbar()

plt.subplot(1,3,2)
plt.pcolormesh(x,y,(ExUMF-Ex0).transpose(),shading="auto")
plt.xlabel("x")
plt.ylabel("y")
plt.title("$E_x$ [V/m]")
plt.colorbar()

plt.subplot(1,3,3)
plt.pcolormesh(x,y,(EyUMF-Ey0).transpose(),shading="auto")
plt.xlabel("x")
plt.ylabel("y")
plt.title("$E_y$ [V/m]")
plt.colorbar()


# TO DO
#    1)  add div(E) to show charge density
#    2)  compute velocity and its divergence
#    3)  anything to be done with analytical approach?


