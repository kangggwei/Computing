# Task A
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#%%
# Task A
dx1 = 0.5
dx2 = 0.05
x1 = np.arange(-5, -2+dx1, dx1)
x2 = np.arange(-2+dx2, 3+dx2, dx2)
x3 = np.arange(3+dx1, 5+dx1, dx1)

x = np.hstack((x1, x2, x3))

y = np.sin(x)

plt.scatter(x, y)

# %%
# Task B
dx = 0.1
dy = 0.1
x = np.arange(-2*np.pi, 2*np.pi + dx, dx)
y = np.arange(-np.pi, 2*np.pi + dy, dy)
(Xg, Yg) = np.meshgrid(x, y)

f = np.sin(Xg) * np.cos(Yg)
g = np.cos(Xg) * np.sin(Yg)

s = f + g
p = f * g

#%%
# Task C Part 1
ax = plt.axes(projection='3d')
ax.plot_surface(Xg, Yg, s)
plt.show()
plt.contour(Xg, Yg, s)
plt.show()
ax = plt.axes(projection='3d')
ax.plot_surface(Xg, Yg, p)
plt.show()
plt.contour(Xg, Yg, p)

#%%
# Task C Part 2
dt = 0.05
t = np.arange(0, 10+dt, dt)

(Xg, Yg, Tg) = np.meshgrid(x, y, t)

r = np.sin(Xg) * np.cos(Yg) * np.exp(-0.5*Tg)

plt.show()
ax = plt.axes(projection='3d')
ax.plot_surface(Xg[:,:,0], Yg[:,:,0], r[:,:,0])

t1 = int(len(t)/2)
plt.show()
ax = plt.axes(projection='3d')
ax.plot_surface(Xg[:,:,t1], Yg[:,:,t1], r[:,:,t1])

# %%
# Task D
dx = 0.5
dy = 0.5

x = np.arange(-5, 5+dx, dx)
y = np.arange(-5, 5+dy, dy)
(Xg, Yg) = np.meshgrid(x, y)

# a)
# fx = Xg
# fy = Yg

# b)
# fx = Yg
# fy = -Xg

# c)
fx = Yg / (Xg ** 2 + Yg ** 2)
fy = -Xg / (Xg ** 2 + Yg ** 2)

plt.streamplot(Xg, Yg, fx, fy)
plt.show()
plt.quiver(Xg, Yg, fx, fy)
plt.show()

#%%
# Task E
dx = 0.1
x = np.arange(-5, 5+dx, dx)
y = np.sin(x)
# ym = abs(y)
ym = np.zeros(len(y))
ym[y<0] = -y[y<0]
ym[y>=0] = y[y>=0]
plt.plot(x, y, 'r')
plt.plot(x,ym, 'b')


ymsat = np.zeros(len(y))
ymsat[ym <= 0.5] = ym[ym <= 0.5]
ymsat[ym > 0.5] = 0.5
ymsat[x <= 0] = 0
plt.plot(x, ymsat, 'm')

# %%
