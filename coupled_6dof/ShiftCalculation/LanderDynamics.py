# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:42:22 2021

@author: omkar_mulekar
"""

import numpy as np
from scipy import integrate


def LanderEOM_ANN(t,x,u):
    
       # States:
    #   x[0]:  x
    #   x[1]:  y
    #   x[2]:  z
    #   x[3]:  dx
    #   x[4]:  dy
    #   x[5]:  dz
    #   x[6]:  phi
    #   x[7]:  theta
    #   x[8]:  psi
    #   x[9]:  p
    #   x[10]: q
    #   x[11]: r

    phi = x[6]
    theta = x[7]
    psi = x[8]
    p = x[9]
    q = x[10]
    r = x[11]

    # Parameters
    g = 9.81
    g0 = 9.81
    Isp = 300



    u1 = np.clip(u[0],0,20)
    u2 = np.clip(u[1],-20,20)
    u3 = np.clip(u[2],-20,20)
    u4 = np.clip(u[3],-20,20)

    
    
    

    dx = x[3]
    dy = x[4]
    dz = x[5]
    ddx = u1*( np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi))
    ddy = u1*(-np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi))
    ddz = u1*( np.cos(phi)*np.cos(theta)) - g
    dphi   = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
    dtheta =     q*np.cos(phi)               - r*np.sin(phi)
    dpsi   =     q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)
    
    Ix = 1.0
    Iy = 0.9
    Iz = 0.8
    dp = (1/Ix)*(u2 + (Iy-Iz)*q*r)
    dq = (1/Iy)*(u3 + (Iz-Ix)*r*p)
    dr = (1/Iz)*(u4 + (Ix-Iy)*p*q)

    dm    = - np.sqrt(u1**2 + u2**2 + u3**2 + u4**2) / (Isp*g0)

    # if (0.9 <= t <= 1.1) or (1.4 <= t <= 1.6):
    #     ddx += 5
    #     ddy -= 5
    #     print('Disturbance Applied!')


    xdot = np.array([dx,dy,dz,ddx,ddy,ddz,dphi,dtheta,dpsi,dp,dq,dr,dm])
    
    return xdot