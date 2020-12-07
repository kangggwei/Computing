# %%
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# %%
# Task A

def simpson(func, A, B, M=1):
  N = 2*M
  h = (B-A)/N
  x = np.linspace(A, B, N+1)
  y = func(x)
  I = y[0] + y[-1]
  for i in range(1, len(y)-1):
    if i % 2 == 0:
      I += 2*y[i]
    else:
      I += 4*y[i]
  return I*h/3

def adaptive(func, A, B, tolerance):
  mid = (A + B)/2
  I_prev = simpson(func, A, B)
  I_curr = simpson(func, A, mid) + simpson(func, mid, B)
  if abs(I_curr - I_prev) > 15*tolerance:
    s = adaptive(func, A, mid, 0.5*tolerance) + adaptive(func, mid, B, 0.5*tolerance)
  else:
    s = I_curr + (I_curr-I_prev)/15
  return s

f = np.sin

simpson(f,0,1,5)
adaptive(f, 0, np.pi/2, 0.00001)
# %%
# Task B

def derivative(k, func, A, B, n):
  x = np.linspace(A, B, n)
  h = (B-A) / (n-1)
  y = func(x)

  ans = [0 for i in range(n-k)]
  i = 0

  while i + k < n:
    for j in range(k+1):
      ans[i] += y[i+k-j]*(-1)**(j)*factorial(k)/factorial(j)/factorial(k-j)
    ans[i] /= h**k
    i += 1

  return ans

derivative(5, np.sin, 0, np.pi, 10)

def plot_derivative(k, func, A, B, n):
  x = np.linspace(A, B, n)
  y = func(x)
  d = derivative(k, func, A, B, n)

  plt.scatter(x, y)
  plt.scatter(x[:len(d)], d)
  plt.show()

plot_derivative(5, np.sin, 0, np.pi, 10)

# %%
# Task C3

def Lagrangian(j, xp, xn):
  res = 1
  xj = xn[j]
  for x in xn:
    if x != xj:
      res *= (xp-x)/(xj-x)
  return res

def interpolate(nodes, A, B):
  xn = np.linspace(A, B, nodes)
  yn = np.sin(xn)
  y_values = np.zeros(nodes)

  for x_index, x in enumerate(xn):
    for index, y in enumerate(yn):
      y_values[x_index] += y * Lagrangian(index, x, xn)
  
  return y_values

def derivative2(k, x, y, n):
  h = x[1] - x[0]

  ans = [0 for i in range(len(x)-k)]
  i = 0

  while i + k < n:
    for j in range(k+1):
      ans[i] += y[i+k-j]*(-1)**(j)*factorial(k)/factorial(j)/factorial(k-j)
    ans[i] /= h**k
    i += 1

  return ans

n = 100
A = 0
B = np.pi
x = np.linspace(A,B,n)
p = interpolate(n,A,B)
d = derivative2(5, x, p, 100)
plt.plot(x, p)
plt.plot(x[:len(d)], d)
plt.show()
# %%
