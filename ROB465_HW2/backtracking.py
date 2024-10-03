import numpy as np

def backtrack(f, grad_f, x, d, alpha = 0.1, beta = 0.6):
    t = 1
    while f(x + t * d) > f(x) + alpha * t * np.dot(grad_f(x) ,d):
        t *= beta
    return t

