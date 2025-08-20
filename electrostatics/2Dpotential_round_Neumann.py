#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 20:31:56 2025

@author: zettergm
"""

###############################################################################
def plot_grad_region(param,x,rho0,drho):
    minEy=np.min(param)
    maxEy=np.max(param)
    plt.plot([rho0,rho0],[minEy,maxEy],'--')
    plt.plot([rho0+drho,rho0+drho],[minEy,maxEy],'--')
    
    plt.plot([-rho0,-rho0],[minEy,maxEy],'--')
    plt.plot([-rho0-drho,-rho0-drho],[minEy,maxEy],'--')    
###############################################################################


###############################################################################
import numpy as np
from potential_parameterize import solve_elliptic_neumann, plot_results
import matplotlib.pyplot as plt

# parameters of problem:
Ey0=-0.05            # background field in which the object is immersed
Ex0=0.0
n0=4e11             # density at center of structure
n1=2e11             # background density
rho0=50e3            # radius of structure along semiminor axis
L=20e3               # gradient scale length at structure edge

semimajor=1.0
rho0maj=rho0*semimajor      # distance from center to edge along semimajor axis
edgedist=4*rho0-rho0        # tests suggest this boundary is sufficiently far away from structure edge along semimajor axis
a=4*rho0;                   # x extent
b=rho0maj + edgedist;       # y extent
lx=256
ly=256

# Run the solve
x,y,Phi,Ex,Ey,n,drho = solve_elliptic_neumann(a,b,semimajor,lx,ly,Ex0,Ey0,rho0,n0,n1,L)
plot_results(x,y,Ex0,Ey0,n,Phi,Ex,Ey,rho0,drho,semimajor)
###############################################################################


###############################################################################
# Compute a velocity field assuming B is in -z direction (NH)
B=50000e-9
vx=-Ey/B
vy=Ex/B

plt.subplots(1,2)

plt.subplot(1,2,1)
plt.pcolormesh(x,y,vx.transpose(),shading='auto')
plt.colorbar()
plt.title('$v_x$')

plt.subplot(1,2,2)
plt.pcolormesh(x,y,vy.transpose(),shading='auto')
plt.colorbar()
plt.title('$v_y$')


# Extract behavior along the y=0 line
Eyctrline=(Ey[:,ly//2]-Ey0)
plt.figure()
plt.plot(x,Eyctrline)
plot_grad_region(Eyctrline,x,rho0,drho)

vxctrline=vx[:,ly//2]
plt.figure()
plt.plot(x,vxctrline)
plot_grad_region(vxctrline,x,rho0,drho)
###############################################################################


