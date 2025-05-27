from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

theta = np.linspace(-4*np.pi, 4*np.pi, 100)
z = np.linspace(-2, 2, 100)

r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.plot(x, y, z, label = 'parametric curve')
ax.scatter(x, y, z)
ax.legend()

ax.set_xlabel('X')
ax.set_xlabel('Y')
ax.set_xlabel('Z')

fig.tight_layout()
plt.show()



fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

_x = np.arange(4)
_y = np.arange(5)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = x + y
bottom = np.zeros_like(top)
width = depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade = True)
ax1.set_title('Shaded')

ax2.bar3d(x, y, bottom, width, depth, top, shade = False)
ax2.set_title('Not Shaded')

plt.show()



ax = plt.figure().add_subplot(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)

ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw = 0.5, rstride = 8, cstride = 8,
                alpha = 0.3)

ax.contour(X, Y, Z, zdir = 'z', offset = -100, cmap = 'coolwarm')
ax.contour(X, Y, Z, zdir = 'z', offset = -40, cmap = 'coolwarm')
ax.contour(X, Y, Z, zdir = 'z', offset = 40, cmap = 'coolwarm')

ax.set(xlim = (-40, 40), ylim = (-40, 40), zlim = (-100, 100),
       xlabel = 'X', ylabel = 'Y', zlabel = 'Z')
plt.show()


