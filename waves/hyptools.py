#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:50:49 2022

@author: zettergm
"""

# imports
import numpy as np

# Implements the Lax-Wendroff method for solving hyperbolic equations.
# Performs a single time update for time step dt.  By default this code
# will assume periodic boundary conditions. This particular code implements
# the two-step LW algorithm.
def LaxWen(dt,dx,v,f):
    lx=f.size
    fleft=np.copy(f[lx-3:lx-1])    #ghost cells here implement periodic boundary conditions
    fright=np.copy(f[0:2])

    #half step lax-f update for cell edges. note indexing for fhalf,
    #i->i-1/2, i+1->i+1/2
    fhalf=np.zeros(lx+1)
    fhalf[0]=1/2*(fleft[1]+f[0])-dt/2/dx*v*(f[0]-fleft[1]);
    for i in range(1,lx):
        fhalf[i]=1/2*(f[i-1]+f[i])-dt/2/dx*v*(f[i]-f[i-1]);
    fhalf[lx]=1/2*(f[lx-1]+fright[0])-dt/2/dx*v*(fright[0]-f[lx-1]);

    #full time step LW update
    fnew=np.zeros(f.size)
    fnew[0]=f[0]-dt/dx*v*(fhalf[1]-fhalf[0])
    for i in range(1,lx-1):
        fnew[i]=f[i]-dt/dx*v*(fhalf[i+1]-fhalf[i])
    fnew[lx-1]=f[lx-1]-dt/dx*v*(fhalf[lx]-fhalf[lx-1])

    return fnew