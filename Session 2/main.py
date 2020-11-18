# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Task A
def MyGauss(A, b):
  n = len(A)
  for i in range(n-1):
    for j in range(i+1, n):
      p = A[j,i] / A[i,i]
      for k in range(i, n):
        A[j,k] -= p * A[i,k]
      b[j] -= p * b[i] 
      
  x = np.zeros(n)
  for i in range(n-1, -1, -1):
    x[i] = b[i] / A[i,i]
    for k in range(i+1, n):
      x[i] -= A[i,k] * x[k] / A[i,i]
  return x

# %%
# Test Task A
A = np.array([[8, -2, 1, 3], [1, -5, 2, 1],[-1, 2, 7, 2],[2 ,-1, 3, 8]], dtype=float)
b = np.array([9, -7, -1, 5], dtype=float)

MyGauss(A, b)
# %%
# Task B.1
dx = 0.1
x = np.arange(-5, 5+dx, dx)
y1 = (2 - 4*x) / 3
y2 = (1 - 8*x) / 6

plt.plot(x, y1)
plt.plot(x, y2)
plt.show()

# %%
# Test Task B.1
A = np.array([[4, 3], [8, 6]], dtype=float)
b = np.array([2, 1], dtype=float)
print(MyGauss(A, b))
np.linalg.det(A)

# %%
# Task B.2
dx = 0.1
x = np.arange(-100, 99+dx, dx)
y1 = (400*x-200) / 201
y2 = (800*x-200) / 401

plt.plot(x, y1)
plt.plot(x, y2)
plt.show()

y3 = (401*x-200) / 201

plt.plot(x, y3)
plt.plot(x, y2)
plt.show()
# %%
A = np.array([[400, -201], [-800, 401]])
b = np.array([200, -200])
x = MyGauss(A, b)
det = np.linalg.det(A)
print(x)
print(det)

C = np.array([[401, -201], [-800, 401]])
x = MyGauss(C, b)
det = np.linalg.det(C)
print(x)
print(det)
# %%
# Task C
# [i1, i2, i3, i4, i5] = [E]
R1 = 50.0
R2 = 100.0
R3 = 50.0
R4 = 100.0
R5 = 50.0
V = 1.0

A = np.array([[R1,-R2,R3,0,0],[1,1,0,-1,-1],[0,R2,0,0,R5],[0,1,1,0,-1],[0,0,R3,-R4,R5]])
b = np.array([0,0,V,0,0])
x = MyGauss(A, b)
det = np.linalg.det(A)
print(x)
print(det)

I = x[0]+x[1]
print(V/I)

# %%
