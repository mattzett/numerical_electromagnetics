#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 11:16:02 2025

@author: zettergm
"""


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
from potential_parameterize_ellipse import solve_elliptic_neumann,plot_results

# Fixed parameters of the solution
Ey0=-0.05            # background field in which the object is immersed
Ex0=0.0
n0=4e11             # density at center of structure
n1=2e11             # background density
a=50e3              # emiminor axis
L=40e3              # gradient scale length at structure edge

ds = np.array([1.1,2,3,4,5,6,7,8,10,12,14,16,20])    # ratio of semimajor to semiminor axis length
#semimajors = np.array([1,4])

Eyctr=np.zeros( (ds.size) )
for i in range(0,ds.size):
    # Variable parameters of problem:
    d=ds[i]
    b=a*d                       # distance from center to edge along semimajor axis
    edgedist=7*a-a        # tests suggest this boundary is sufficiently far away from structure edge along semimajor axis
    xmax=7*a;                   # x extent
    ymax=b + edgedist;       # y extent

    lx=128                      # number of grid points along semiminor axis direction (x)  
    dyref=2*xmax/np.real(lx)    # reference cell spacing
    ly=int(np.ceil(2*ymax/dyref))  # number of grid points along semimajor axis direction (y)
    
    print("Running with a,b = ", a,b, " and lx,ly = ", lx,ly)
    x,y,Phi,Ex,Ey,n,ddist = solve_elliptic_neumann(xmax,ymax,lx,ly,a,b,Ex0,Ey0,n0,n1,L)
    plot_results(x,y,Ex0,Ey0,n,Phi,Ex,Ey,a,b)
    Eyctr[i]=(Ey-Ey0)[lx//2,ly//2]      # take only the residual (polarization) field


plt.figure()
plt.plot(ds*a/L,Eyctr/Eyctr[0])
plt.xlabel('$c/\ell$')
plt.ylabel('$|E_y/E_{y,round}|$')

###############################################################################
###############################################################################