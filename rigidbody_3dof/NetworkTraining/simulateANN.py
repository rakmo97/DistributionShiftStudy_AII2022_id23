
#%% ============================================================================
# Import Dependencies
# ============================================================================
from scipy.io import loadmat
from scipy.io import savemat
from tensorflow.keras import models
import pickle
import numpy as np
from scipy import integrate
import LanderDynamics as LD
import time

from matplotlib import pyplot as plt
# %matplotlib inline


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
filename = base_data_folder+formulation+'NetworkTraining/ANN2_703_tanh_n75.h5'
policy = models.load_model(filename)

nState    =   7
nCtrl     =   2


# Time settings
t0 = 0
tf = 3.7
nt = 500
times = np.linspace(t0,tf,nt)


# Trajectory settings
trajToRun = 1
starting_idx = trajToRun*100
x_ocl = matfile['Xtest2'].reshape(-1,nState)[starting_idx:starting_idx+100,:]
u_ocl = matfile['ttest2'][starting_idx:starting_idx+100,:]
times_ocl = matfile['times_test'][starting_idx:starting_idx+100,:]

x0 = x_ocl[0,:]
y_policy = np.zeros([nt,nCtrl])
x_policy = np.zeros([nt,nState])
x_policy[0,:] = x0

print(x0)

tic = time.perf_counter()
print('Running Sim')
for i in range(nt-1):
    
    y_policy[i,:] = policy.predict(x_policy[i,:].reshape(1,-1))
    # print('Predicted Control {} of {}'.format(i,nt))
    
    # sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM(t,y,y_policy[i,:]),\
    #                                 t_span=(times[i],times[i+1]), \
    #                                 y0=x_policy[i,:]) # Default method: rk45
        
    sol = integrate.solve_ivp(fun=lambda t, y: LD.LanderEOM_ANN(t,y,policy),\
                                    t_span=(times[i],times[i+1]), \
                                    y0=x_policy[i,:]) # Default method: rk45
    # print('Simulated step {} of {}'.format(i,nt))


    xsol = sol.y
    tsol = sol.t
                
    
    x_policy[i+1,:] = xsol[:,xsol.shape[1]-1]
    # print('State: {}'.format(x_policy[i+1,:]))
    # print('State norm: {}'.format(np.linalg.norm(x_policy[i+1,:])))

    if np.linalg.norm(x_policy[i+1,:]) < 1.0:
        print('Target reached, breaking out of sim')
        break
    # x_policy[i+1,:] = x_ocl[i+1,:]

x_policy = x_policy[:i+1,:]
y_policy = y_policy[:i+1,:]
times = times[:i+1]

runtime = time.perf_counter() - tic

print('Average prediction time: {} s'.format(runtime/nt))

y_policy = policy.predict(x_policy)

mdic = {"x_policy": x_policy,
        "y_policy": y_policy,
        "times": times}
savemat("matlab_matrix.mat", mdic)
    
    
    
# ============================================================================
# Plotting
# ============================================================================


plt.figure(1)
plt.plot(x_ocl[:,0],x_ocl[:,1])
plt.plot(x_policy[:,0],x_policy[:,1],'--')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['OpenOCL','ANN'],loc='best')
plt.savefig('ann_traj.png')


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
plt.savefig('ann_pos_mass.png')




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
plt.savefig('ann_vel.png')

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
plt.savefig('ann_ctrls.png')


