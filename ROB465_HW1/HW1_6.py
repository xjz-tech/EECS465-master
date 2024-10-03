import numpy as np
A = np.array([[1, 2], [3, -1]])
B = np.array([[-2, -2], [4, -3]])
C1 = A + 2 * B    # A+2B
C2 = np.dot(A, B) # AB
C3 = np.dot(B, A) # BA
C4 = np.transpose(A)
C5 = np.dot(B, B)
C6 = np.dot(np.transpose(A), np.transpose(B))
C7 = np.transpose(np.dot(A, B))
C8 = np.linalg.det(A)
C9 = np.linalg.inv(B)

print('A + 2B' + '\n', C1)
print('AB' + '\n',C2)
print('BA' + '\n',C3)
print('AT' + '\n',C4)
print('B*B' + '\n',C5)
print('ATBT' + '\n',C6)
print('(AB)T' + '\n',C7)
print('det(A)' + '\n',C8)
print('inverse B' + '\n',C9)



















