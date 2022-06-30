
#%% ============================================================================
# Import Dependencies
# ============================================================================


import pickle
from scipy.io import loadmat
from scipy.io import savemat
from KNNregressor import KNNReg
import numpy as np
from scipy import integrate
import LanderDynamics as LD
import time
from matplotlib import pyplot as plt


#%% ============================================================================
# Load Trajectories
# ===========================================================================
print("Loading mat file")
# base_data_folder = 'E:/Research_Data/3DoF_RigidBody/'
base_data_folder = '/orange/rcstudents/omkarmulekar/DistributionShiftStudy/'
formulation = 'rigidbody_3dof/'
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')
# matfile = loadmat('ANN2_decoupled_data.mat')

#%% ============================================================================
# Load Policy
# ============================================================================
print('Loading Policy')
# filename = 'BC_k2_w_end.pkl'
filename = 'BC_k2.pkl'
with open(filename,'rb') as f:
    pickle_in = pickle.load(f)
    
knn_interp = pickle_in[0]

nState    =   knn_interp.xData.shape[1]
nCtrl     =   knn_interp.yData.shape[1]





# Trajectory settings
trajToRun = 1
starting_idx = trajToRun*100
x_ocl = matfile['Xtest2'].reshape(-1,nState)[starting_idx:starting_idx+100,:]
u_ocl = matfile['ttest2'][starting_idx:starting_idx+100,:]
times_ocl = matfile['times_test'][starting_idx:starting_idx+100,:]

# Time settings
t0 = 0
tf = times_ocl[-1]
nt = 200
times = np.linspace(t0,tf,nt)

x0 = knn_interp.x_test_data[starting_idx,:]
# x0 = np.array([ -1200,  1000, -8.93259019e-02,  8.12677630e+00, -5.63516203e-03, -1.17370117e-01])

y_policy = np.zeros([nt,nCtrl])
x_policy = np.zeros([nt,nState])
x_policy[0,:] = x0

tic = time.perf_counter()
print('Running Sim')
for i in range(nt-1):
    
    
    y_policy[i,:] = knn_interp.predict(x_policy[i,:])

    print("==============================================")
    print('Predicted Control {} of {}'.format(i,nt))
    
    sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM_KNN(t,y,y_policy[i,:]),\
                                    t_span=(times[i],times[i+1]), \
                                    y0=x_policy[i,:]) # Default method: rk45

    print('Simulated step {} of {}'.format(i,nt))


    xsol = sol.y
    tsol = sol.t
                
    
    x_policy[i+1,:] = xsol[:,xsol.shape[1]-1]
    # x_policy[i+1,:] = x_ocl[i+1,:]

runtime = time.perf_counter() - tic

print('Average prediction time: {} s'.format(runtime/nt))

#%% ============================================================================
# Plotting
# ============================================================================

   
# ============================================================================
# Plotting
# ============================================================================


plt.figure(1)
plt.plot(x_ocl[:,0],x_ocl[:,1])
plt.plot(x_policy[:,0],x_policy[:,1],'--')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['OpenOCL','ANN'],loc='best')
plt.savefig('knn_traj.png')


plt.figure(2)
plt.subplot(221)
plt.plot(times_ocl,x_ocl[:,0])
plt.plot(times,x_policy[:,0],'--')
plt.ylabel('x [m]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(222)
plt.plot(times_ocl,x_ocl[:,1])
plt.plot(times,x_policy[:,1],'--')
plt.ylabel('y [m]')

plt.subplot(223)
plt.plot(times_ocl,x_ocl[:,2]*180.0/np.pi)
plt.plot(times,x_policy[:,2]*180.0/np.pi,'--')
plt.xlabel('Time [s]')
plt.ylabel('phi [deg]')

plt.subplot(224)
plt.plot(times_ocl,x_ocl[:,6])
plt.plot(times,x_policy[:,6],'--')
plt.xlabel('Time [s]')
plt.ylabel('m [kg]')

plt.suptitle('Trajectory',y=1.05)
plt.tight_layout()
plt.savefig('knn_pos_mass.png')




plt.figure(3)
plt.subplot(221)
plt.plot(times_ocl,x_ocl[:,3])
plt.plot(times,x_policy[:,3],'--')
plt.ylabel('x-dot [m/s]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(222)
plt.plot(times_ocl,x_ocl[:,4])
plt.plot(times,x_policy[:,4],'--')
plt.ylabel('y-dot [m/s]')

plt.subplot(223)
plt.plot(times_ocl,x_ocl[:,5]*180.0/np.pi)
plt.plot(times,x_policy[:,5]*180.0/np.pi,'--')
plt.xlabel('Time [s]')
plt.ylabel('phi-dot [deg/s]')
plt.savefig('knn_vel.png')

plt.suptitle('Trajectory',y=1.05)
plt.tight_layout()




plt.figure(4)
plt.subplot(211)
plt.plot(times_ocl,u_ocl[:,0])
plt.plot(times,y_policy[:,0],'--')
plt.ylabel('Fx [N]')
plt.legend(['OpenOCL','ANN'],loc='best')

plt.subplot(212)
plt.plot(times_ocl,u_ocl[:,1])
plt.plot(times,y_policy[:,1],'--')
plt.ylabel('Fy [N]')
plt.savefig('knn_ctrls.png')




