import SGDtest
import random
def stochastic_gradient_descent(step_size = 1, iterations = 1000):
    x0 = -5
    x_sequence = [x0]
    for it in range(iterations):
        index = random.randint(0, 10000-1)
        gradient = SGDtest.fiprime(x0, index)
        x = x0 - step_size * gradient
        x0 = x
        x_sequence.append(x0)
    return x_sequence

