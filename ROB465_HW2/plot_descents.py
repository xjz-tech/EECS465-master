import numpy as np
import matplotlib.pyplot as plt
from gradientdescent import gradient_descent
from newtonsmethod import newtons_method

def f(x):
    return np.exp(0.5 * x + 1) + np.exp(-0.5 * x - 0.5) + 5 * x

def grad_f(x):
    return 0.5 * np.exp(0.5 * x + 1) - 0.5 * np.exp(-0.5 * x - 0.5) + 5

def hessian_f(x):
    return 0.25 * np.exp(0.5 * x + 1) + 0.25 * np.exp(-0.5 * x - 0.5)
x0 = 5
x_axis = np.linspace(-10, 10, 200)
y_axis = [f(x) for x in x_axis
          ]

gd_result, point_gd, fgd_sequence = gradient_descent(f, grad_f, x0, )
nt_result, point_nt, fnt_sequence = newtons_method(f, grad_f, hessian_f, x0)
# print(gd_result)
# print(nt_result)
plt.figure(figsize=(12, 10))

plt.plot(x_axis, y_axis, label='Objective Function', color='black')
print(point_nt.shape)
print(fnt_sequence.shape)
plt.scatter(point_gd, fgd_sequence, color='red', label='Gradient Descent', s = 30)
plt.scatter(point_nt, fnt_sequence, color='magenta', label='Newton\'s Method', s = 30)
plt.xlabel('x')
plt.ylabel('f(x)')

plt.title('Objective Function and Optimization Results')
plt.savefig('Optimization Results.png')
plt.legend()

plt.figure(figsize=(12, 10))

plt.plot(np.arange(0, point_gd.shape[0]), fgd_sequence, 'red', label='Gradient Descent')
plt.legend()
plt.plot(np.arange(0, point_nt.shape[0]), fnt_sequence, 'magenta', label='Newton\'s Method')
plt.title('Comparation between Newtons Method and Gradient Descent')

plt.xlabel('iterations')
plt.ylabel('f(x)')
plt.savefig('Comparation between Newtons Method and Gradient Descent.png')
plt.legend()
plt.show()