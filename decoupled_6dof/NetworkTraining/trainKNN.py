# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:22:37 2020

@author: Omkar
"""

from KNNregressor import KNNReg
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
import pickle
import time
from matplotlib import pyplot as plt


#%% ============================================================================
# Load Data
# ============================================================================
# Load in training and testing data
print("Loading mat file")
base_data_folder = '/orange/rcstudents/omkarmulekar/DistributionShiftStudy/'
formulation = 'decoupled_6dof/'
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')
saveflag = ''

ctrls_all_list = matfile['tfull_2']
ctrls_train_list = matfile['ttrain2']
ctrls_test_list = matfile['ttest2']
states_all_list = matfile['Xfull_2']
states_train_list = matfile['Xtrain2']
states_test_list = matfile['Xtest2']
times_all_list = matfile['times']
times_train_list = matfile['times_train']
times_test_list = matfile['times_test']


#%% ============================================================================
# Create KNN regressor
# ============================================================================
print('Creating KNN')
K = 9
knn_interp = KNNReg(states_train_list, ctrls_train_list, k=K,
                    t_train_data=times_train_list, 
                    x_test_data=states_test_list, y_test_data=ctrls_test_list, t_test_data=times_test_list)

#%% ============================================================================
# Open Loop Test
# ============================================================================
# num_test = ctrls_test_list.shape[0]
num_test = 100
print('Open Loop Testing on {} points'.format(num_test))

y_test = np.empty(ctrls_test_list.shape)

tic = time.perf_counter()
for i in range(num_test):
    
    y_test[i,:] = knn_interp.predict(states_test_list[i,:])
    
runtime_openloop = time.perf_counter() - tic

print("Average prediction time: {} s".format(runtime_openloop/num_test))

#%% ============================================================================
# Plot Open Loop Test
# ============================================================================

plt.figure()
plt.subplot(211)
plt.plot(y_test[:num_test,0])
plt.plot(ctrls_test_list[:num_test,0],'--')
plt.ylabel('Fx [N]')
plt.legend(['Predicted','Optimal'])
plt.subplot(212)
plt.plot(y_test[:num_test,1])
plt.plot(ctrls_test_list[:num_test,1],'--')
plt.ylabel('Fy [N]')
plt.tight_layout()
plt.savefig('KNN_OL_test.png')

#%% ============================================================================
# Save KNN
# ============================================================================
saveoff_list = [knn_interp]
saveoff_list.append(['knn_interp'])

with open('BC_k2.pkl','wb') as f:
    pickle.dump(saveoff_list, f)
print('Saved')