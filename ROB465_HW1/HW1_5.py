import numpy as np

a1 = np.array([[0, 0, -1],
              [4, 1,  1],
              [-2, 2,  1]])

b1 = np.array([3,1,1])
try:
    x1 = np.linalg.solve(a1, b1)
    print(x1)
except:
    print(" NO SOLUTIONS!")

a2 = np.array([[0, -2, 6],[-4, -2, -2], [2, 1, 1]])
b2 = np.array([1, -2, 0])
try:
    x2 = np.linalg.solve(a2, b2)
    print(x2)
except:
    print("NO SOLUTIONS!")

a3 = np.array([[2, -2], [-4, 3]])
b3 = np.array([3, -2])
try:
    x3 = np.linalg.solve(a3, b3)
    print(x3)
except:
    print("NO SOLITIONS!")



