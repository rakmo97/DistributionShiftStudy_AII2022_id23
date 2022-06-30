
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
formulation = 'decoupled_6dof/'
matfile = loadmat(base_data_folder+formulation+'ANN2_data.mat')
# matfile = loadmat('ANN2_decoupled_data.mat')

#%% ============================================================================
# Load Policy
# ============================================================================
print('Loading Policy')
filename = base_data_folder+formulation+'NetworkTraining/ANN2_703_tanh_n250.h5'
policy = models.load_model(filename)

nState    =   13
nCtrl     =   4


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

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_policy[:,0],x_policy[:,1],x_policy[:,2])
ax.plot(x_ocl[:,0],x_ocl[:,1],x_ocl[:,2],'--')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend(['Policy','Optimal'])
plt.savefig('ann_traj.png')


plt.figure()
plt.subplot(221)
plt.plot(times,x_policy[:,0])
plt.plot(times_ocl,x_ocl[:,0],'--')
plt.xlabel('Time [s]')
plt.ylabel('X [m]')
plt.legend(['Policy','Optimal'])

plt.subplot(222)
plt.plot(times,x_policy[:,1])
plt.plot(times_ocl,x_ocl[:,1],'--')
plt.xlabel('Time [s]')
plt.ylabel('Y [m]')
plt.legend(['Policy','Optimal'])

plt.subplot(223)
plt.plot(times,x_policy[:,2])
plt.plot(times_ocl,x_ocl[:,2],'--')
plt.xlabel('Time [s]')
plt.ylabel('Z [m]')
plt.legend(['Policy','Optimal'])

plt.subplot(224)
plt.plot(times,x_policy[:,12])
plt.plot(times_ocl,x_ocl[:,12],'--')
plt.xlabel('Time [s]')
plt.ylabel('m [kg]')
plt.legend(['Policy','Optimal'])
plt.tight_layout()
plt.savefig('ann_pos.png')




plt.figure()
plt.subplot(221)
plt.plot(times,x_policy[:,3])
plt.plot(times_ocl,x_ocl[:,3],'--')
plt.xlabel('Time [s]')
plt.ylabel('Xdot [m/s]')
plt.legend(['Policy','Optimal'])

plt.subplot(222)
plt.plot(times,x_policy[:,4])
plt.plot(times_ocl,x_ocl[:,4],'--')
plt.xlabel('Time [s]')
plt.ylabel('Ydot [m/s]')
plt.legend(['Policy','Optimal'])

plt.subplot(223)
plt.plot(times,x_policy[:,5]*180.0/np.pi)
plt.plot(times_ocl,x_ocl[:,5]*180.0/np.pi,'--')
plt.xlabel('Time [s]')
plt.ylabel('Zdot [m/s]')
plt.legend(['Policy','Optimal'])
plt.tight_layout()
plt.savefig('ann_vel.png')



plt.figure()
plt.subplot(221)
plt.plot(times,x_policy[:,6]*180.0/np.pi)
plt.plot(times_ocl,x_ocl[:,6]*180.0/np.pi,'--')
plt.xlabel('Time [s]')
plt.ylabel('Phi [deg]')
plt.legend(['Policy','Optimal'])

plt.subplot(222)
plt.plot(times,x_policy[:,7]*180.0/np.pi)
plt.plot(times_ocl,x_ocl[:,7]*180.0/np.pi,'--')
plt.xlabel('Time [s]')
plt.ylabel('Theta [deg]')
plt.legend(['Policy','Optimal'])

plt.subplot(223)
plt.plot(times,x_policy[:,8]*180.0/np.pi)
plt.plot(times_ocl,x_ocl[:,8]*180.0/np.pi,'--')
plt.xlabel('Time [s]')
plt.ylabel('Psi [deg]')
plt.legend(['Policy','Optimal'])
plt.tight_layout()
plt.savefig('ann_eulerangles.png')



plt.figure()
plt.subplot(221)
plt.plot(times,x_policy[:,9]*180.0/np.pi)
plt.plot(times_ocl,x_ocl[:,9]*180.0/np.pi,'--')
plt.xlabel('Time [s]')
plt.ylabel('p [deg/s]')
plt.legend(['Policy','Optimal'])

plt.subplot(222)
plt.plot(times,x_policy[:,10]*180.0/np.pi)
plt.plot(times_ocl,x_ocl[:,10]*180.0/np.pi,'--')
plt.xlabel('Time [s]')
plt.ylabel('q [deg/s]')
plt.legend(['Policy','Optimal'])

plt.subplot(223)
plt.plot(times,x_policy[:,11]*180.0/np.pi)
plt.plot(times_ocl,x_ocl[:,11]*180.0/np.pi,'--')
plt.xlabel('Time [s]')
plt.ylabel('r [deg/s]')
plt.legend(['Policy','Optimal'])
plt.tight_layout()
plt.savefig('ann_angvel.png')



plt.figure()
plt.subplot(221)
plt.plot(times,y_policy[:,0])
plt.plot(times_ocl,u_ocl[:,0],'--')
plt.xlabel('Time [s]')
plt.ylabel('u1 [N]')
plt.legend(['Policy','Optimal'])

plt.subplot(222)
plt.plot(times,y_policy[:,1])
plt.plot(times_ocl,u_ocl[:,1],'--')
plt.xlabel('Time [s]')
plt.ylabel('u2 [rad/s/s]')
plt.legend(['Policy','Optimal'])

plt.subplot(223)
plt.plot(times,y_policy[:,2])
plt.plot(times_ocl,u_ocl[:,2],'--')
plt.xlabel('Time [s]')
plt.ylabel('u3 [rad/s/s]')
plt.legend(['Policy','Optimal'])

plt.subplot(224)
plt.plot(times,y_policy[:,3])
plt.plot(times_ocl,u_ocl[:,3],'--')
plt.xlabel('Time [s]')
plt.ylabel('u4 [rad/s/s]')
plt.legend(['Policy','Optimal'])
plt.tight_layout()
plt.savefig('ann_ctrls.png')
