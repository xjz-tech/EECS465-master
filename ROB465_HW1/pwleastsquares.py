import numpy as np
import matplotlib.pyplot as plt

# data
data = np.loadtxt('calibration.txt')
command = data[:, 0]  # command
measured = data[:, 1]  # measured

# knot points
knots = [-0.5, 0.5]


A = np.zeros((len(command), 4))
" We write it this way because it is a continuous function, continuous at the connection points"
" , and each time new bias and coefficients need to be introduced for fitting."
A[:, 0] = command
A[:, 1] = np.maximum(0, command + 0.5)
A[:, 2] = np.maximum(0, command - 0.5)
# bias
A[:, 3] = 1
# print(A.shape)
# Compute the pseudo-inverse
A_pinv = np.linalg.pinv(A)
# print(A_pinv.shape)
# Compute the parameters
params = A_pinv.dot(measured)      #(4,)
# print(params.shape)
# print(A)
# Compute the fitted piecewise linear function
fit = A.dot(params)

# print(y_fit.shape)

# Compute the sum of squared errors
sse = np.sum((measured - fit)**2)

print("Piecewise linear parameters (a1, a2, a3, b):")
print(f"a1 = {params[0]:.3f}")
print(f"a2 = {params[1]:.3f}")
print(f"a3 = {params[2]:.3f}")
print(f"b = {params[3]:.3f}")
print(f"Sum of squared errors: {sse:.3f}")

# Prediction for 0.68
x_pred = 0.68
y_pred = params[0] * 0.68 + params[1] * np.maximum(0, 0.68 + 0.5)  + params[2] * np.maximum(0, 0.68 - 0.5) + params[3]
print(f"Prediction for x = 0.68: y = {y_pred:.3f}")
# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(command, measured, color='blue', marker='x', label='Data')  # x axis --- measured ; y axis --- commanded
plt.plot(command, fit, color='red', label='Piecewise Linear') # x axis -- measured; y axis --
plt.xlabel('Commanded Position')
plt.ylabel('Measured Position')
plt.title('Piecewise Linear Least Squares Fit')
plt.savefig('pwleast_squares_fit.png')
plt.legend()
plt.show()

