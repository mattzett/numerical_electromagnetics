#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 10:41:07 2025

@author: zettergm
"""



###############################################################################
#  Solver
###############################################################################
def solve_elliptic_neumann(xmax,ymax,lx,ly,a,b,Ex0,Ey0,n0,n1,L):
    # imports
    import numpy as np
    import scipy
    from elliptical_coords import cart2elliptical,elliptical_metric
    
    # create a 2D grid
    x=np.linspace(-xmax,xmax,lx)
    y=np.linspace(-ymax,ymax,ly)
    dx=x[1]-x[0]
    dy=y[1]-y[0]
    [X,Y]=np.meshgrid(x,y,indexing='ij')
    N = lx*ly

    # auxiliary parameters of reference ellipse, for convenience    
    c=np.sqrt(b**2-a**2)     # elliptic eccentricity (b>a)
    d=np.sqrt(b**2/a**2)     # ratio of semimajor to semiminor axes of ellipse

    # elliptical coordinates on our grid using reference ellipse of semiminor
    #   semimajor size a,b
    #MU,NU = cart2elliptical(X,Y,a,b)  
    MU,NU = cart2elliptical(Y,X,a,b)    # rotate ellipse 90 degrees (semimajor aligned y)
        
    # spherical coordinates on grid (if needed)
    #rho=np.sqrt(X**2+Y**2)
    #phi=np.arctan2(Y,X)
    
    # Neumann boundary condition for four sides of square, y sides are dictated
    #   by inputs Ey0 to this function
    #f1=np.zeros( (lx) )
    #f2=np.zeros( (lx) )
    potmax=-Ey0*(y[-1]-y[0])      # potential at grid edge
    g1=np.linspace(0,potmax,ly)
    g2=g1
    
    # ODE solution for distance from beginning location of gradient where the value
    #   of n1 density is reached.  This solution is only valid along the semiminor
    #   axis (a) but we are attempting to scale this to any location below when we
    #   compute density on the grid.
    dist1=a-L*np.log(n1/n0)    # solution for end of gradient region given a starting point and scale length
    ddist=dist1-a              # distance across gradient region along semiminor axis
    hmu,hnu = elliptical_metric(MU, NU, a, b)
        
    # reference value of elliptic coordinate corresponding to our desired ellipse
    muref=np.arcsinh(np.sqrt(a**2/c**2))
    #muref2=np.arccosh(np.sqrt(b**2/c**2))     # should be the same???
        
    n = np.empty( (lx,ly) )
    gradcells=0.0
    numgrad=0
    for i in range(0,lx):
        for j in range (0,ly):
            nuhere=NU[i,j]    # angular position (ell. coords.) of this point wrt reference ellipse
            # the edge of the reference ellipse is at nuhere,mu=1; 
            
            # We need to compute a dmu corresponding to the distance ddist in order
            #   to know bounds for applying gradient
            dmu = ddist/hmu[i,j]
            
            if MU[i,j] < muref:
                n[i,j]=n0
            elif MU[i,j] >= muref and MU[i,j] < muref+dmu: 
                n[i,j]=n0*np.exp(-(MU[i,j]-muref)/(L/hmu[i,j]))    # ODE solution for density of fixed scale length
                gradcells+=hmu[i,j]*dmu/dy
                numgrad+=1
            else:
                n[i,j]=n1          

    if (numgrad>0):
        gradcells/=numgrad
        
    # Ballpark the number of cells in the gradient region
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
    
    return x,y,PhiUMF,ExUMF,EyUMF,n,ddist
###############################################################################


###############################################################################
# Draw a reference ellipse on a plot
###############################################################################
def drawedge(a,b):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Draw a circle on the inner and outer limits of the gradient region
    xcir=np.linspace(-a,a,128)
    ycir=np.sqrt(b**2/a**2)*np.sqrt(a**2-xcir**2)
    ycir2=-np.sqrt(b**2/a**2)*np.sqrt(a**2-xcir**2)      # other half of solution (neg. sqrt)
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
def plot_results(x,y,Ex0,Ey0,n,Phi,Ex,Ey,a,b):
    import matplotlib.pyplot as plt
    
    plt.subplots(1,4,figsize=(14,6),dpi=100)
    
    plt.subplot(1,4,1)
    plt.pcolormesh(x,y,n.transpose(),shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sparse LU factorization [V]")
    plt.colorbar()
    drawedge(a,b)
    
    plt.subplot(1,4,2)
    plt.pcolormesh(x,y,Phi.transpose(),shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sparse LU factorization [V]")
    plt.colorbar()
    drawedge(a,b)
    
    plt.subplot(1,4,3)
    plt.pcolormesh(x,y,(Ex-Ex0).transpose(),shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$E_x$ [V/m]")
    plt.colorbar()
    drawedge(a,b)
    
    plt.subplot(1,4,4)
    plt.pcolormesh(x,y,(Ey-Ey0).transpose(),shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$E_y$ [V/m]")
    plt.colorbar()
    drawedge(a,b)

    return
###############################################################################

