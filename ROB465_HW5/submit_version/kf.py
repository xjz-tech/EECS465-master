import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle

#this plots the covariance matrix as an ellipsoid at 2*sigma
def plot_cov(mean,cov,plot_axes):
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    ell = Ellipse(xy=mean,
              width=lambda_[0]*2, height=lambda_[1]*2,
              angle=np.rad2deg(np.arccos(v[0, 0])))
    #ell.set_facecolor('none')
    ell.set_facecolor((1.0, 1.0, 1.0, 0))
    ell.set_edgecolor((0, 0, 0, 1))
    plot_axes.add_artist(ell)
    plt.scatter(mean[0,0],mean[1,0],c='r',s=5)


# Implement the Kalman filter in this function
def KalmanFilter(mu, Sigma, z, u, A, B, C, Q, R):
    ### YOUR CODE HERE ###
    # Prediction Step
    mu_bar = A @ mu + B @ u
    Sigma_bar = A @ Sigma @ A.T + B @ R @ B.T
    # Sigma_bar = A @ Sigma @ A.T + B @ Q @ B.T

    # Correction Step
    K = Sigma_bar @ C.T @ np.linalg.inv(C @ Sigma_bar @ C.T + Q)
    mu_new = mu_bar + K @ (z - C @ mu_bar)
    Sigma_new = (np.eye(len(mu)) - K @ C) @ Sigma_bar
    ### YOUR CODE HERE ###
    return mu_new, Sigma_new

def main():
    #initialize the figure to draw stuff
    plt.ion()
    plot_axes = plt.subplot(111, aspect='equal')   

    #load in the data
    PIK = "kfdata.dat"
    with open(PIK, "rb") as f:
        noisy_measurement,actions,ground_truth_states,N = pickle.load(f)

    #your model parameters are imported here
    from kfmodel import A, B, C, Q, R

    #initialize the mean of the state estimate guassian
    mu = np.matrix(noisy_measurement[:,0]).transpose()

    #initialize the covariance of the state estimate guassian, Sigma
    Sigma = np.eye(2)

    ###YOUR CODE HERE###
    #specify number of data points to consider
    #it may be easier to debug with fewer data points
    #remember to set this back to N = 100 to get the screenshot for your pdf
    N = 100
    ###YOUR CODE HERE###

    #go through each measurement and action...
    #and estimate the state using the Kalman filter
    estimated_states = np.zeros((2,N))
    for i in range(1,N):
        z = np.matrix(noisy_measurement[:,i]).transpose() #current x
        u = np.matrix(actions[:,i]).transpose()           #current u
        #run the Kalman Filter
        mu, Sigma = KalmanFilter(mu, Sigma, z, u, A, B, C, Q, R); 
        #store the result
        estimated_states[:,i] = np.squeeze(mu)  
        #draw covariance every 3 steps (drawing every step is too cluttered)
        if i%3==0:
            plot_cov(mu,Sigma,plot_axes)

    #compute the error between your estimate and ground truth
    state_errors = estimated_states[:,0:N] - ground_truth_states[:,0:N]
    total_error=np.sum(np.linalg.norm(state_errors, axis=0))
    print("Total Error: %f"%total_error)

    #draw the data and result
    plt.plot(ground_truth_states[0,0:N], ground_truth_states[1,0:N],'b',linewidth=1.0)
    plt.scatter(noisy_measurement[0,0:N], noisy_measurement[1,0:N],c='k',s=2)
    plt.plot(estimated_states[0,0:N], estimated_states[1,0:N],'r',linewidth=1.0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.pause(.001)
    plt.ioff()
    plt.show()
  
    #input("Press enter to exit")  

if __name__ == '__main__':
    main()
