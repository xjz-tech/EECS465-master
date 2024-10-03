import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('calibration.txt')
Measure = data[:, 1]   # measured data
Command = data[:, 0]   # commanded data
A = np.column_stack((Command, np.ones_like(Command)))
A_pinv = np.linalg.pinv(A)   # measured data A+
params = A_pinv.dot(Measure)       # x (2,)
y_fit = A.dot(params)        #
sse = np.sum((Measure - y_fit)**2)
print(f"Params:  slope = {params[0]:.2f}, intercept = {params[1]:.2f}")
print(f"squared errors: {sse:.3f}")

plt.figure(figsize=(10, 6))
plt.scatter(Command, Measure, color='blue', marker='x', label='Data')
plt.plot(Command, y_fit, color='red', label='Line')
plt.xlabel('Commanded Position')
plt.ylabel('Measured Position')
plt.title('Least Squares Fit')
plt.savefig('least_squares_fit.png')
plt.legend()
plt.show()


