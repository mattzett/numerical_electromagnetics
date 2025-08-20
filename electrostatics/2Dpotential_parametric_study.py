#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 16:40:34 2025

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
from potential_parameterize import solve_elliptic_neumann,plot_results

# Fixed parameters of the solution
Ey0=-0.05            # background field in which the object is immersed
Ex0=0.0
n0=4e11             # density at center of structure
n1=2e11             # background density
rho0=50e3            # radius of structure along semiminor axis
L=20e3               # gradient scale length at structure edge

semimajors = np.array([1,2,3,4,5,6,7,8,10,12,14,16,20])    # ratio of semimajor to semiminor axis length
#semimajors = np.array([1,4])

Eyctr=np.zeros( (semimajors.size) )
for i in range(0,semimajors.size):
    # Variable parameters of problem:
    semimajor=semimajors[i]
    rho0maj=rho0*semimajor      # distance from center to edge along semimajor axis
    edgedist=7*rho0-rho0        # tests suggest this boundary is sufficiently far away from structure edge along semimajor axis
    a=7*rho0;                   # x extent
    b=rho0maj + edgedist;       # y extent

    lx=128                      # number of grid points along semiminor axis direction (x)  
    dyref=2*a/np.real(lx)       # reference cell spacing
    ly=int(np.ceil(2*b/dyref))  # number of grid points along semimajor axis direction (y)
    
    print("Running with a,b = ", a,b, " and lx,ly = ", lx,ly)
    x,y,Phi,Ex,Ey,n,drho = solve_elliptic_neumann(a,b,semimajor,lx,ly,Ex0,Ey0,rho0,n0,n1,L)
    plot_results(x,y,Ex0,Ey0,n,Phi,Ex,Ey,rho0,drho,semimajor)
    Eyctr[i]=(Ey-Ey0)[lx//2,ly//2]      # take only the residual (polarization) field


plt.figure()
plt.plot(semimajors*rho0/L,Eyctr/Eyctr[0])
plt.xlabel('$c/\ell$')
plt.ylabel('$|E_x/E_{x,round}|$')

###############################################################################
###############################################################################