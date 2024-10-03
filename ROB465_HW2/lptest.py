import numpy as np
import cvxpy as cp

A = np.array([[ 0.7071,  0.7071],
              [-0.7071,  0.7071],
              [ 0.7071, -0.7071],
              [-0.7071, -0.7071]])

b = np.array([1.5, 1.5, 1, 1])
obj_coef = np.array([2, 1])
x = cp.Variable(2)
objective = cp.Minimize(obj_coef.T @ x)
constraints = [A @ x <= b]
problem = cp.Problem(objective, constraints)
problem.solve()
print("The optimal point is: x1 = {:.2f}, x2 = {:.2f}".format(x.value[0], x.value[1]))

