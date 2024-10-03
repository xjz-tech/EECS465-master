import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

position = np.loadtxt('calibration.txt', delimiter=' ')

commanded = position[:, 0].reshape(-1, 1)   # get commanded position
A = np.hstack((commanded, np.ones(commanded.shape)))   # add column of 1
measured = position[:, 1].reshape(-1, 1)   # get measured position

x = np.linalg.inv(A.T @ A) @ A.T @ measured   # psuedo-inverse
predict = commanded * x[0] + x[1]   # predict position
SSE = np.sum((predict - measured)**2)   # sum of squared error

print("The parameter values:", np.array2string(x, prefix="The parameter values: "))   # parameters [x1, x2]
print("The sum of squared errors: ", np.array2string(SSE))

# plot
plt.scatter(commanded, measured, color="blue", label = "Commanded vs measured")
plt.plot(commanded, predict, color="red", label = "Fitted line")
plt.legend()
plt.show()