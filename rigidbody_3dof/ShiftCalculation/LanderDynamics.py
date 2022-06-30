# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:42:22 2021

@author: omkar_mulekar
"""

import numpy as np
from scipy import integrate


def LanderEOM_ANN(t,x,u):
    
    # States:
    #   x[0]: x
    #   x[1]: y
    #   x[2]: phi
    #   x[3]: dx
    #   x[4]: dy
    #   x[5]: dphi
    #   x[6]: m
    
    # Parameters
    
    g = 9.81
    g0 = 9.81
    Isp = 300

    phi = x[2]
    m = x[6]


    Fx = np.clip(u[0],-20,20)
    Fy = np.clip(u[1],0,20)
    
    
    dx    = x[3]
    dy    = x[4]
    dphi  = x[5]
    ddx   = (1/m)*(Fx*np.cos(phi) - Fy*np.sin(phi))
    ddy   = (1/m)*(Fx*np.sin(phi) + Fy*np.cos(phi)) - g
    ddphi  = (1/m)*Fx
    dm    = - np.sqrt(Fx**2 + Fy**2) / (Isp*g0)

    xdot = np.array([dx,dy,dphi,ddx,ddy,ddphi,dm])
    
    
    return xdot