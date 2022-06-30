# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:42:22 2021

@author: omkar_mulekar
"""
import sys
sys.path.append('../NetworkTraining')
import math

import numpy as np
from scipy import integrate
import LanderDynamics as LD
from sklearn.mixture import GaussianMixture

# ============================================================================
def CalculateKLDivergenceNormals(Pdata,Qdata):
    
    muP = np.mean(Pdata,axis=0)
    covP = np.cov(Pdata.T)
    muQ = np.mean(Qdata,axis=0)
    covQ = np.cov(Qdata.T)
    
    d = Pdata.shape[1]
    
    KLDivergence = 0.5*(np.trace(np.linalg.inv(covQ)@covP) + (muQ-muP).T@np.linalg.inv(covQ)@(muQ-muP) - d + np.log(np.linalg.det(covQ)/np.linalg.det(covP)))
    
    return KLDivergence

# ============================================================================
def CalculateKLDivergenceNormals_fromMeanCov(muP, covP, muQ, covQ):
    
    
    d = covP.shape[1]
    
    KLDivergence = 0.5*(np.trace(np.linalg.inv(covQ)@covP) + (muQ-muP).T@np.linalg.inv(covQ)@(muQ-muP) - d + np.log(np.linalg.det(covQ)/np.linalg.det(covP)))
    
    return KLDivergence

# ============================================================================


def CalculateKLDivergenceGMMs_Variational(Pdata,Qdata, numComponents=4, n_trials=10):
    KLDivergencelist = np.zeros(n_trials)
    
    for i in range(n_trials):
        print("i = {}".format(i))
        gmP = GaussianMixture(n_components=numComponents, covariance_type='full').fit(Pdata)
        gmQ = GaussianMixture(n_components=numComponents, covariance_type='full').fit(Qdata)
           
        KLDivergence_upperbound = 0
        for a in range(numComponents):
            for b in range(numComponents):
                KLDivergence_upperbound += gmP.weights_[a]*gmQ.weights_[b]*CalculateKLDivergenceNormals_fromMeanCov(gmP.means_[a],gmP.covariances_[a],gmQ.means_[b],gmQ.covariances_[b])
        
        KLDivergence = 0
        for a in range(numComponents):
            numerator = 0
            for ap in range(numComponents):
                numerator += gmP.weights_[ap]*np.exp(-CalculateKLDivergenceNormals_fromMeanCov(gmP.means_[a],gmP.covariances_[a],gmP.means_[ap],gmP.covariances_[ap]))
                
            denominator = 0
            for b in range(numComponents):
                denominator += gmQ.weights_[b]*np.exp(-CalculateKLDivergenceNormals_fromMeanCov(gmP.means_[a],gmP.covariances_[a],gmQ.means_[b],gmQ.covariances_[b]))
                
            KLDivergence += gmP.weights_[a]*np.log(numerator/denominator)
        
        KLDivergencelist[i] = KLDivergence
        print(KLDivergence)
    
    KLDivergence = KLDivergencelist.min()
    return KLDivergence, KLDivergence_upperbound


# ============================================================================
def CalculateGMMProbabilityDensity(gm,Xin):
    
    k = gm.means_[0].shape[0]
    
    probability_density = 0
    for i in range(gm.n_components):
        probability_density += gm.weights_[i]*(np.exp(-0.5*(Xin-gm.means_[i]).T@np.linalg.inv(gm.covariances_[i])@(Xin-gm.means_[i]))/(np.sqrt(((2*np.pi)**k)*np.linalg.det(gm.covariances_[i]))))
    
    
    return probability_density

# ============================================================================
def RunPolicyWithBeta_ANN(x0,x_OCL,u_OCL,t_OCL,policy,beta):
    
    nt = t_OCL.shape[0]
    nCtrl = u_OCL.shape[1]
    nState = x_OCL.shape[1]
    
    y_policy = np.zeros([nt,nCtrl])
    x_policy = np.zeros([nt,nState])
    x_policy[0,:] = x0

    Fapplied = np.zeros(u_OCL.shape)

    for j in range(nt-1):
        
        y_policy[j,:] = policy.predict(x_policy[j,:].reshape(1,-1))
        
        F_input = (beta)*u_OCL[j,:] + (1-beta)*y_policy[j,:]
        Fapplied[j,:] = F_input
        

        sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM_ANN(t,y,F_input),\
                                        t_span=(t_OCL[j],t_OCL[j+1]), \
                                        y0=x_policy[j,:]) # Default method: rk45
            

    
        xsol = sol.y
        tsol = sol.t
                    
        
        x_policy[j+1,:] = xsol[:,xsol.shape[1]-1]



    
    return x_policy, Fapplied, beta
    
