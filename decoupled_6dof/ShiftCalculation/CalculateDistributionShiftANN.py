# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:42:22 2021

@author: omkar_mulekar
"""

# ============================================================================
# Import Dependencies
# ============================================================================
from tensorflow.keras import models
print('models loaded')
# import ImitationLearningFunctions as ILF
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import time
from matplotlib import pyplot as plt
# %matplotlib inline
import DistributionShiftFunctions as DSF
from scipy import integrate

#%% ============================================================================
# Load Trajectories
# ===========================================================================
print("Loading mat file")
# base_data_folder = 'E:/Research_Data/DistributionShiftStudy/'
base_data_folder = '/orange/rcstudents/omkarmulekar/DistributionShiftStudy/'
formulation = 'decoupled_6dof/'
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')

ctrlsOCLfull = matfile['tfull_2']
times_full = matfile['times']
trajsOCLfull =  matfile['Xfull_2']

traj_plotting = False

#%% ============================================================================
# Load Policy
# ============================================================================
print('Loading Policy')
filename = base_data_folder+formulation+'NetworkTraining/ANN2_703_tanh_n250.h5'
policy = models.load_model(filename)

print("Controller Loaded")



# ============================================================================
# Beta Iteration Loop
# ============================================================================
num_betas = 10
num_trajs = 100
num_states = 13

num_pairs = num_trajs*100

Pdata = np.zeros([num_pairs,num_states])
Qbetadata = np.zeros([num_betas,num_pairs,num_states])
D_KL_beta = np.zeros([num_betas])
D_KL_beta_upperbound = np.zeros([num_betas])

# Set up Pdata array
Pdata = trajsOCLfull[:num_pairs,:]



# Values of Beta to simulate
betas = np.linspace(0,1,num=num_betas)

# Run sims and set up Qbeta Data arrays
for i in range(num_betas):
# for i in range(1):

    print("+===============================")

    beta = betas[i]
    print("Running Trajs for Beta = {}".format(beta))

    count = 0
    
    for j in range(num_trajs):
    # for j in range(1):
        
        if (j % 5 == 0):
            print("Trajectory {} of {}".format(j,num_trajs))
        
        trajs_idx = j*100
        t_OCL = times_full[trajs_idx:trajs_idx+100]
        u_OCL = ctrlsOCLfull[trajs_idx:trajs_idx+100,:]
        x_OCL = trajsOCLfull[trajs_idx:trajs_idx+100,:]
        x0 = x_OCL[0]
        
        # Run sim
        states, Fapplied, beta = DSF.RunPolicyWithBeta_ANN(x0,x_OCL,u_OCL,t_OCL,policy,beta)
        
        for k in range(100):
            Qbetadata[i,count,:] = states[k]
            count += 1
        
            
        
    # Get kl divergence for current Qbeta data
    # D_KL_beta[i] = DSF.CalculateKLDivergenceNormals(Pdata,Qbetadata[i])
    D_KL_beta[i], D_KL_beta_upperbound[i] = DSF.CalculateKLDivergenceGMMs_Variational(Pdata,Qbetadata[i])
    print("Beta: {}, D_KL: {}, D_KL_upper: {}".format(beta, D_KL_beta[i], D_KL_beta_upperbound[i]))
    

plt.figure()
plt.plot(betas,D_KL_beta)
plt.xlabel("Beta")
plt.ylabel("D_KL(beta)")
plt.savefig('ANN_DKL_Beta.png')



M_pi = integrate.trapz(D_KL_beta, betas)
print("Shift Metric: {}".format(M_pi))

mdic = {"betas": betas,
        "D_KL_beta": D_KL_beta,
        "D_KL_beta_upperbound": D_KL_beta_upperbound,
        "Pdata": Pdata,
        "Qbetadata": Qbetadata}
matfilename = formulation[:-1] + '_ANN_shift_calc.mat'

savemat(matfilename, mdic)





