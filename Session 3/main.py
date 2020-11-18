# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# %%
# Task A

f = lambda x :1/np.sqrt(x**20.10 + 2020)

def trapzeqd(b, func):
  nodes = 5
  x = np.linspace(0, b, nodes)
  y = func(x)
  interval = x[1]
  integral = (y[0] + y[-1])/2
  for i in range(1, len(y)-1):
    integral += y[i]
  return integral*interval

ans = trapzeqd(1.75,f)
print(ans)

b = [10, 100, 1000, 10000]
y = list(map(lambda x: trapzeqd(x,f), b))
plt.plot(b, y)
plt.show()

def trapzeqd2(b, func):
  interval = 0.5
  x = np.arange(0, b+interval, interval)
  y = func(x)
  integral = (y[0] + y[-1])/2
  for i in range(1, len(y)-1):
    integral += y[i]
  return integral*interval

y2 = list(map(lambda x: trapzeqd2(x,f), b))
plt.scatter(b, y2)
plt.show()


# %%
# Task B

g = lambda x :1/np.sqrt(x**1.10 + 2020)

y3 = list(map(lambda x: trapzeqd(x,g), b))
plt.plot(b, y3)
plt.show()

y4 = list(map(lambda x: trapzeqd2(x,f), b))
plt.scatter(b, y4)
plt.show()

# %%
# Task C

def trapz(x, y):
  integral = 0
  for i in range(len(x)-1):
    interval = x[i+1] - x[i]
    integral += interval*(y[i] + y[i+1])/2
  return integral

# %%
# Task D


def read_file(filename):
  with open(filename, 'r') as f:
    f = f.readlines()
  return list(map(float, f))

xs = read_file('xs.txt')
xn = read_file('xn.txt')
ys = read_file('ys.txt')
yn = read_file('yn.txt')

plt.plot(xs, ys)
plt.plot(xn, yn)
plt.axis('equal')
plt.show()

surface = trapz(xn, yn) - trapz(xs, ys)
print(surface*1.0e-6)

# %%
# Task E
R = 5
dx = 0.05

x = np.arange(-R+dx, R, dx)
N = len(x)

G = np.zeros(N)

for i in range(N):
  mx = np.sqrt(5**2-x[i]**2)

  y = np.arange(-mx+dx, mx, dx)
  z = np.zeros(len(y))

  for j in range(len(y)):
    z[j] = np.sqrt(R-np.sqrt(x[i]**2+y[j]**2))

  G[i] = trapz(y, z)

I = trapz(x, G)

print(I)
# %%
r = np.linspace(0,R,100)
t = np.linspace(0,2*np.pi,100)
# set 2D mesh grids
[Rg, Tg] = np.meshgrid(r,t)

# calculate X and Y (2D meshgrids)
X = Rg*np.cos(Tg)
Y = Rg*np.sin(Tg)

# calculate Z(X,Y)
Z = np.sqrt(R-np.sqrt(X**2+Y**2))

# plost as surface
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,Z)
# %%
