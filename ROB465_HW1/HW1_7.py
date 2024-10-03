import numpy as np
from numpy import cos as cos
from numpy import sin as sin
from numpy import pi as pi
R1 = np.array([[0, -1, 0],[1, 0, 0], [0, 0, 1]])
R2 = np.array([[cos(pi / 5), 0, sin(pi / 5)],
               [0,           1,           0],
               [-sin(pi / 5),0, cos(pi / 5)]])
R3 = np.array([[-1, 0, 0],
               [0, -1, 0],
               [0, 0,  1]])
R = np.dot(R3, np.dot(R2, R1) )
print(R)