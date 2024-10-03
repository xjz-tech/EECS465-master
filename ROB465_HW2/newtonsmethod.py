import numpy as np
from backtracking import backtrack
# def newtons_method(f, grad_f, hessian_f, x0, epsilon = 0.0001, alpha = 0.1, beta = 0.6):
#     x = x0
#     while True:
#         print(hessian_f(x))
#         hessian_inv = np.linalg.inv(hessian_f(x))
#         d = -hessian_inv @ grad_f(x)
#         newtons_decrement = np.dot(grad_f(x).T, hessian_f @ grad_f(x))
#         step_size = backtrack(f(x), grad_f(x), x, d, alpha, beta)
#         x_next = x + step_size * d
#         if (newtons_decrement / 2) < epsilon:
#             return x_next
#         x = x_next
def newtons_method(f, grad_f, hessian_f, x0, epsilon = 0.0001, alpha = 0.1, beta = 0.6):
    x = x0

    point_sequence = [x0]
    f_sequnce = [f(x)]
    while True:
        hessian = hessian_f(x)
        hessian_inv = 1 / hessian
        d = -hessian_inv * grad_f(x)
        newtons_decrement = grad_f(x) * hessian_inv * grad_f(x)
        if (newtons_decrement / 2) < epsilon:
            return x, np.array(point_sequence, dtype='object'), np.array(f_sequnce)
        step_size = backtrack(f, grad_f, x, d, alpha, beta)
        x_next = x + step_size * d

        f_sequnce.append(f(x_next))
        point_sequence.append(x_next)

        x = x_next