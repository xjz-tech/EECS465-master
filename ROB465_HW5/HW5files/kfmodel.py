import numpy as np

###YOUR CODE HERE###
#motion model
A = np.matrix([
    [1.0, 0.0],
    [0.0, 1.0]
]) #replace with your A

B = np.matrix([
    [1.5, 0.1],
    [0.2, -0.5]
]) #replace with your B

#sensor model
C = np.matrix([
    [1.05, 0.01],
    [0.01, 0.90]
]) #replace with your C
    
#motion noise covariance
R =np.matrix([
 [2.50696845e-03,1.79957758e-05],
 [1.79957758e-05, 2.51063277e-03]
])  #replace with your R

#sensor noise covariance
Q = np.matrix([
[ 0.04869528, -0.0058636 ],
[-0.0058636 ,  1.01216104]
]) #replace with your Q
###YOUR CODE HERE###
