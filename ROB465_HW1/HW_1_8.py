import numpy as np
print(np.cos(np.pi + np.arctan(4/7)))
print(np.sin(np.pi + np.arctan(4/7)))
A = np.array([[0, 0, -1],
              [-0.496, -0.868, 0],
              [-0.868, 0.496, 0]])
print(np.linalg.det(A))