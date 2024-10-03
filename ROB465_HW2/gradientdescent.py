import numpy as np
from backtracking import backtrack
def gradient_descent(f, grad_f, x0, epsilon = 0.0001, alpha = 0.1, beta = 0.6):
    x = x0
    point_sequence = [x]
    f_sequence = [f(x)]
    while True:
        d = -grad_f(x)
        if np.linalg.norm(d) < epsilon:
            return x, np.array(point_sequence), np.array(f_sequence)

        step_size = backtrack(f, grad_f, x, d, alpha, beta)
        x_next = x + step_size * d
        point_sequence.append(x_next)
        f_sequence.append(f(x_next))
        x = x_next

