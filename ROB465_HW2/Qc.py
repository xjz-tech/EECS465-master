import numpy as np
import random

maxi = 10000

def fi(x, i):
    coef1 = 0.01 + (0.5 - 0.01) * i / maxi
    coef2 = 1 + (6 - 1) * i / maxi
    return (np.exp(coef1 * x + 0.1) + np.exp(-coef1 * x - 0.5) - coef2 * x) / (maxi / 100)

def fiprime(x, i):
    coef1 = 0.01 + (0.5 - 0.01) * i / maxi
    coef2 = 1 + (6 - 1) * i / maxi
    return (coef1 * np.exp(coef1 * x + 0.1) - coef1 * np.exp(-coef1 * x - 0.5) - coef2) / (maxi / 100)

def fsum(x):
    return sum(fi(x, i) for i in range(maxi))

# SGD function
def stochastic_gradient_descent(step_size=1, iterations=1000):
    x0 = -5
    for _ in range(iterations):
        index = random.randint(0, maxi - 1)
        gradient = fiprime(x0, index)
        x0 -= step_size * gradient
    return x0

num_runs = 30
results_1000 = []

for _ in range(num_runs):
    x_final = stochastic_gradient_descent(iterations=1000)
    final_fsum = fsum(x_final)
    results_1000.append(final_fsum)

mean_1000 = np.mean(results_1000)
variance_1000 = np.var(results_1000)

results_750 = []

for _ in range(num_runs):
    x_final = stochastic_gradient_descent(iterations=750)
    final_fsum = fsum(x_final)
    results_750.append(final_fsum)

mean_750 = np.mean(results_750)
variance_750 = np.var(results_750)

print("1000 Iterations - Mean:", mean_1000, "Variance:", variance_1000)
print("750 Iterations - Mean:", mean_750, "Variance:", variance_750)