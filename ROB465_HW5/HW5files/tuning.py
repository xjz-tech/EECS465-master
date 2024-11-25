#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle

if __name__ == '__main__':
    
    #initialize plotting        
    plt.ion()
    
    #load in the data
    PIK = "kfdata.dat"
    with open(PIK, "rb") as f:
        noisy_measurement,actions,ground_truth_states,N = pickle.load(f)

    #your model parameters are imported here
    from kfmodel import A, B, C, Q, R

    #we are assuming both the motion and sensor noise is 0 mean
    motion_errors = np.zeros((2,N))
    sensor_errors = np.zeros((2,N))
    for i in range(1,N):
        x_t = np.matrix(ground_truth_states[:,i]).transpose()
        x_tminus1 = np.matrix(ground_truth_states[:,i-1]).transpose()
        u_t = np.matrix(actions[:,i]).transpose()
        z_t = np.matrix(noisy_measurement[:,i]).transpose()
        ###YOUR CODE HERE###
        # Compute the predicted state based on the motion model
        x_pred = A @ x_tminus1 + B @ u_t
        # Calculate the motion error
        motion_error = x_t - x_pred
        motion_errors[:, i] = np.squeeze(np.asarray(motion_error))

        # Compute the predicted measurement based on the sensor model
        z_pred = C @ x_t
        # Calculate the sensor error
        sensor_error = z_t - z_pred
        sensor_errors[:, i] = np.squeeze(np.asarray(sensor_error))
        ###YOUR CODE HERE###
    
    motion_cov=np.cov(motion_errors)
    sensor_cov=np.cov(sensor_errors)
    
    print("Motion Covariance:")
    print(motion_cov)
    print("Measurement Covariance:")
    print(sensor_cov)

    
