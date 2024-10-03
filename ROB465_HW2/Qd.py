import numpy as np
import time
import random
from newtonsmethod import newtons_method
from gradientdescent import gradient_descent





# Define functions
maxi = 10000

def fi(x, i):
    coef1 = 0.01 + (0.5 - 0.01) * i / maxi
    coef2 = 1 + (6 - 1) * i / maxi
    return (np.exp(coef1 * x + 0.1) + np.exp(-coef1 * x - 0.5) - coef2 * x) / (maxi / 100)

def fiprime(x, i):
    coef1 = 0.01 + (0.5 - 0.01) * i / maxi
    coef2 = 1 + (6 - 1) * i / maxi
    return (coef1 * np.exp(coef1 * x + 0.1) - coef1 * np.exp(-coef1 * x - 0.5) - coef2) / (maxi / 100)

def fiprimeprime(x, i):
    coef1 = 0.01 + (0.5 - 0.01) * i / maxi
    return (coef1**2 * np.exp(coef1 * x + 0.1) + coef1**2 * np.exp(-coef1 * x - 0.5)) / (maxi / 100)

def fsum(x):
    return sum(fi(x, i) for i in range(maxi))

def fsumprime(x):
    sum = 0
    for i in range(0,maxi):
       sum = sum + fiprime(x,i)
    return sum

def fsumprimeprime(x):
    sum = 0
    for i in range(0,maxi):
       sum = sum + fiprimeprime(x,i)
    return sum

# SGD implementation
def stochastic_gradient_descent(step_size=1, iterations=1000):
    x = -5
    for _ in range(iterations):
        index = random.randint(0, maxi - 1)
        gradient = fiprime(x, index)
        x -= step_size * gradient
    return x

# Gradient Descent implementation

# Newton's Method implementation


# Timing and comparison
start_time = time.time()
sgd_result = stochastic_gradient_descent()
sgd_time = time.time() - start_time

start_time = time.time()
gd_result, _, _ = gradient_descent(fsum, fsumprime, -5)
gd_time = time.time() - start_time

start_time = time.time()
newton_result, _, _ = newtons_method(fsum, fsumprime, fsumprimeprime, -5)
newton_time = time.time() - start_time

print("SGD Time:", sgd_time)
print("Gradient Descent Time:", gd_time)
print("Newton's Method Time:", newton_time)

print("SGD Result:", fsum(sgd_result))
print("Gradient Descent Result:", fsum(gd_result))
print("Newton's Method Result:", fsum(newton_result))