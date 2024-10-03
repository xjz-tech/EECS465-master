import numpy as np
import matplotlib.pyplot as plt

# 创建坐标数组
a = np.arange(-450, 500, 50)
b = np.arange(-175, 525, 50)

# 生成网格
X, Y = np.meshgrid(a, b)
print(X.shape)
# 创建图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制网格
ax.plot_wireframe(X, Y, np.zeros_like(X), color='blue')

# 设置标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 显示图形
plt.show()
