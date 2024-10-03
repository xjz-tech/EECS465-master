import numpy as np
from sgd import stochastic_gradient_descent
import SGDtest

import matplotlib.pyplot as plt
y = []

x = stochastic_gradient_descent()
for xi in x:
    y.append(SGDtest.fsum(xi))

print('*' * 50)
plt.figure(figsize=(12, 10))
plt.plot(np.arange(0, len(x)), y, label='fsum', color='black')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('stochastic_gradient_descent')
plt.grid()
plt.savefig('stochastic_gradient_descent.png', dpi=300, bbox_inches='tight')
plt.show()

