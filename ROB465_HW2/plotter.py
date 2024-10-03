import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.exp(0.5*x + 1) + np.exp(-0.5*x - 0.5) + 5*x

xvals = np.arange(-10, 10, 0.01) # Grid of 0.01 spacing from -10 to 10
yvals = f(xvals) # Evaluate function on xvals
plt.plot(xvals, yvals) # Create line plot with yvals against xvals

plt.show() #show the plot