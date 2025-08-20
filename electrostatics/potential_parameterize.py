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
    #rho=np.sqrt(X**2+Y**2/semimajor**2)
    rho=np.sqrt(X**2+Y**2)
    phi=np.arctan2(Y,X)
    
    # Neumann boundary condition for four sides of square
    #f1=np.zeros( (lx) )
    #f2=np.zeros( (lx) )
    potmax=-Ey0*(y[-1]-y[0])
    g1=np.linspace(0,potmax,ly)
    g2=g1
    
    rho1=rho0-L*np.log(n1/n0)    # solution for end of gradient region given a starting point and scale length
    drho=rho1-rho0
    n = np.zeros( (lx,ly) )
    for i in range(0,lx):
        for j in range (0,ly):
            # get an ellipse edge distance for this angle-location
            rho0thisphi=np.sqrt(rho0**2*semimajor**2/
                                (semimajor**2*np.cos(phi[i,j])**2 +
                                np.sin(phi[i,j])**2))
            
            if rho[i,j] < rho0thisphi:
                n[i,j]=n0
            elif rho[i,j] > rho0thisphi and rho[i,j] < rho0thisphi+drho: 
                n[i,j]=n0*np.exp(-(rho[i,j]-rho0thisphi)/L)    # ODE solution for density of fixed scale length
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
    
    return x,y,PhiUMF,ExUMF,EyUMF,n,drho
###############################################################################


###############################################################################
# Draw a reference circle on a plot
###############################################################################
def drawedge(rho0,drho,semimajor):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Draw a circle on the inner and outer limits of the gradient region
    xcir=np.linspace(-rho0,rho0,128)
    ycir=semimajor*np.sqrt(rho0**2-xcir**2)
    ycir2=-semimajor*np.sqrt(rho0**2-xcir**2)      # other half of solution (neg. sqrt)
    plt.plot(np.concatenate( (xcir,np.flip(xcir)) ),np.concatenate( (ycir,np.flip(ycir2)) ))
    #xcir=np.linspace(-(rho0+drho),(rho0+drho),128)
    #ycir=semimajor*np.sqrt((rho0+drho)**2-xcir**2)
    #ycir2=-semimajor*np.sqrt((rho0+drho)**2-xcir**2)
    #plt.plot(np.concatenate( (xcir,np.flip(xcir)) ),np.concatenate( (ycir,np.flip(ycir2)) ))

    return
###############################################################################


###############################################################################
# Plotting
###############################################################################
def plot_results(x,y,Ex0,Ey0,n,Phi,Ex,Ey,rho0,drho,semimajor):
    import matplotlib.pyplot as plt
    
    plt.subplots(1,4,figsize=(14,6),dpi=100)
    
    plt.subplot(1,4,1)
    plt.pcolormesh(x,y,n.transpose(),shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sparse LU factorization [V]")
    plt.colorbar()
    drawedge(rho0,drho,semimajor)
    
    plt.subplot(1,4,2)
    plt.pcolormesh(x,y,Phi.transpose(),shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sparse LU factorization [V]")
    plt.colorbar()
    drawedge(rho0,drho,semimajor)
    
    plt.subplot(1,4,3)
    plt.pcolormesh(x,y,(Ex-Ex0).transpose(),shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$E_x$ [V/m]")
    plt.colorbar()
    drawedge(rho0,drho,semimajor)
    
    plt.subplot(1,4,4)
    plt.pcolormesh(x,y,(Ey-Ey0).transpose(),shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$E_y$ [V/m]")
    plt.colorbar()
    drawedge(rho0,drho,semimajor)

    return
###############################################################################

