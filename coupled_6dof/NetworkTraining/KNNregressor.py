import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist 

# %matplotlib inline

class KNNReg:
    
    def __init__(self, xData, yData, k=1, t_train_data=[], x_test_data=[], y_test_data=[], t_test_data=[]):
        
        self.xData = xData
        self.yData = yData
        self.t_train_data = t_train_data
        
        self.x_test_data = x_test_data
        self.y_test_data = y_test_data
        self.t_test_data = t_test_data
                
        self.numOutputs = yData.shape[1]
        self.k = k
    
        self.S = np.cov(xData.T)
        self.Si = np.linalg.inv(self.S)
    

    def predict(self, state_in):
        
        k = self.k
        
        state_data = self.xData
        ctrl_data = self.yData
        
        
        S = np.cov(state_data.T)
        Si = np.linalg.inv(S)
        distances = np.sqrt(np.einsum('ji,jk,ki->i',(state_in-state_data).T,Si,(state_in-state_data).T))


        

        
        
        if k==1:
            low_i = np.argmin(distances)   
            ctrl_out = ctrl_data[low_i]
        else:
            kSmallest_i = np.argpartition(distances,k)[:k]
            kSmallest = ctrl_data[kSmallest_i]
   
            if (distances[kSmallest_i] < 1e-5).any():
                ctrl_out = kSmallest[0]
            else:
                weights_for_avg = 1.0/distances[kSmallest_i]
                ctrl_out = np.average(kSmallest,axis=0, weights=weights_for_avg)


        return ctrl_out
            
