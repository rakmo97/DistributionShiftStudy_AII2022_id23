# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:42:22 2021

@author: omkar_mulekar
"""

import numpy as np
from scipy import integrate


def LanderEOM_ANN(t,x,policy):
    
    # States:
    #   x[0]: x
    #   x[1]: y
    #   x[2]: z
    #   x[3]: dx
    #   x[4]: dy
    #   x[5]: dz
    #   x[6]: m
    
    # Parameters
    
    g = 9.81
    g0 = 9.81
    Isp = 300


    m = x[6]

    u = policy.predict(x.reshape(1,-1))[0]
    Fx = np.clip(u[0],-20,20)
    Fy = np.clip(u[1],-20,20)
    Fz = np.clip(u[2],  0,20)
    

    
    dx    = x[3]
    dy    = x[4]
    dz    = x[5]
    ddx   = (1/x[6])*Fx
    ddy   = (1/x[6])*Fy
    ddz   = (1/x[6])*Fz - g
    dm    = - np.sqrt(Fx**2 + Fy**2 + Fz**2) / (Isp*g0)

    xdot = np.array([dx,dy,dz,ddx,ddy,ddz,dm])
    
    
    return xdot




def LanderEOM_KNN(t,x,u):
    
    # States:
    #   x[0]: x
    #   x[1]: y
    #   x[2]: z
    #   x[3]: dx
    #   x[4]: dy
    #   x[5]: dz
    #   x[6]: m
    
    # Parameters
    
    g = 9.81
    g0 = 9.81
    Isp = 300


    m = x[6]

    Fx = np.clip(u[0],-20,20)
    Fy = np.clip(u[1],-20,20)
    Fz = np.clip(u[2],  0,20)
    

    
    dx    = x[3]
    dy    = x[4]
    dz    = x[5]
    ddx   = (1/x[6])*Fx
    ddy   = (1/x[6])*Fy
    ddz   = (1/x[6])*Fz - g
    dm    = - np.sqrt(Fx**2 + Fy**2 + Fz**2) / (Isp*g0)

    xdot = np.array([dx,dy,dz,ddx,ddy,ddz,dm])
    
    
    return xdot