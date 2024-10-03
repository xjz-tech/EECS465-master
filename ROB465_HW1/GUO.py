import numpy as np
import matplotlib.pyplot as plt
# Load data from file
data = np.loadtxt('calibration.txt')
# Load data from file
Commanded_data = data[:,0]
Measured_data = data[:,1]

# Piece-wise linear least-squares fit
knotpoints = [-0.5, 0.5]

# Create design matrix
B = np.zeros((Commanded_data.shape[0], 4))
B[:,0] = 1
B[:,1] = Commanded_data
B[:,2] = np.maximum(Commanded_data - knotpoints[0], 0)
B[:,3] = np.maximum(Commanded_data - knotpoints[1], 0)

BTB = np.dot(B.T, B)
B_pseudo_inv = np.linalg.pinv(B)

# Least-squares fit
parameters = np.dot(B_pseudo_inv, Measured_data)
print(parameters)

# calculate fitted values
fitted_values = np.dot(B, parameters)
#print(fitted_values)

# Sum of Squared Errors
squared_errors = (Measured_data - fitted_values) ** 2
sum_of_squared_errors = np.sum(squared_errors)

plt.scatter(Commanded_data, Measured_data, color='blue', label='data', marker='x')
plt.plot(Commanded_data, fitted_values, color='red', label='fit_line')
plt.xlabel('Commanded Data')
plt.ylabel('Measured Data')
plt.legend()
plt.title('Least Squares Fit')
plt.grid()
plt.show()

# print('The Slope is:', slope)
# print('The Intercept is:', intercept)
print('The Sum of Squared Errors is:', sum_of_squared_errors)

# Predicted value for 0.68
predicted_value = parameters[0] + parameters[1] * 0.68 + parameters[2] * np.maximum(0.68 - knotpoints[0], 0) + parameters[3] * np.maximum(0.68 - knotpoints[1], 0)
print('The predicted value for 0.68 is:', predicted_value)