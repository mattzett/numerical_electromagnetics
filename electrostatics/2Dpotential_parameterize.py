#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 11:45:21 2025

@author: zettergm
"""

###############################################################################
#  Solver
###############################################################################
def solve_elliptic_neumann(a,b,semimajor,lx,ly,Ex0,Ey0,rho0,n0,n1,L):
    # imports
    import numpy as np
    import scipy
    
    # create a 2D grid
    x=np.linspace(-a,a,lx)
    y=np.linspace(-b,b,ly)
    dx=x[1]-x[0]
    dy=y[1]-y[0]
    [X,Y]=np.meshgrid(x,y,indexing='ij')
    N = lx*ly

    # cylindrical coordinates
    rho=np.sqrt(X**2+Y**2/semimajor**2)
    #phi=np.arctan2(Y,X)
    
    # Neumann boundary condition for four sides of square
    #f1=np.zeros( (lx) )
    #f2=np.zeros( (lx) )
    potmax=-Ey0*(y[-1]-y[0])
    g1=np.linspace(0,potmax,ly)
    g2=g1
    
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
    PhiUMF=np.reshape(PhiUMF, (lx,ly), order='F')
    [ExUMF,EyUMF]=np.gradient(-PhiUMF,x,y)
    
    return x,y,PhiUMF,ExUMF,EyUMF,n
###############################################################################



###############################################################################
# Plotting
###############################################################################
def plot_results(x,y,Ex0,Ey0,Phi,Ex,Ey):
    import matplotlib.pyplot as plt
    
    plt.subplots(1,3,figsize=(14,6),dpi=100)
    plt.subplot(1,3,1)
    plt.pcolormesh(x,y,Phi.transpose(),shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sparse LU factorization [V]")
    plt.colorbar()
    
    plt.subplot(1,3,2)
    plt.pcolormesh(x,y,(Ex-Ex0).transpose(),shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$E_x$ [V/m]")
    plt.colorbar()
    
    plt.subplot(1,3,3)
    plt.pcolormesh(x,y,(Ey-Ey0).transpose(),shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$E_y$ [V/m]")
    plt.colorbar()
    
    return
###############################################################################



###############################################################################
###############################################################################
##   Main Program
###############################################################################
###############################################################################

# As a convergence study we need to take care to keep dx, dy constant to preserve 
#   accuracy in the gradient region and to also keep the boundaryh a fixed distance
#   from the gradient regions in order to avoid having boundaries affect solutions
#   (in a comparative sense, at least).  

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Fixed parameters of the solution
Ey0=-0.1            # background field in which the object is immersed
Ex0=0.0
n0=4e11             # density at center of structure
n1=1e11             # background density
rho0=0.3            # radius of structure along semiminor axis
L=0.1               # gradient scale length at structure edge

#semimajors = np.array([1,2,4,6,8,10])     # >1 makes semimajor axis in y-direction
semimajors = np.array([1,2,3,4,5,6,7,8,10,12,14,16,20])
Eyctr=np.zeros( (semimajors.size) )
for i in range(0,semimajors.size):
    # Variable parameters of problem:
    semimajor=semimajors[i]
    rho0maj=rho0*semimajor      # distance from center to edge along semimajor axis
    edgedist=2.-rho0            # tests suggest this boundary is sufficiently far away from structure edge along semimajor axis
    a=2.0;                      # x extent
    b=rho0maj + edgedist;       # y extent

    lx=128                      # number of grid points along semiminor axis direction (x)  
    dyref=2*2.0/np.real(lx)           # reference cell spacing
    ly=int(np.ceil(2*b/dyref))  # number of grid points along semimajor axis direction (y)
    
    print("Running with a,b = ", a,b, " and lx,ly = ", lx,ly)
    x,y,Phi,Ex,Ey,_ = solve_elliptic_neumann(a,b,semimajor,lx,ly,Ex0,Ey0,rho0,n0,n1,L)
    plot_results(x,y,Ex0,Ey0,Phi,Ex,Ey)
    Eyctr[i]=(Ey-Ey0)[lx//2,ly//2]      # take only the residual (polarization) field


plt.figure()
plt.plot(semimajors/L,Eyctr/Eyctr[0])
plt.xlabel('semimajor axis scaled to gradient-length (unitless)')
plt.ylabel('center field magnitude relative to circular patch (unitless)')


###############################################################################
###############################################################################

