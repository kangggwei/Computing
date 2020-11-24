# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Task A

def Lagrangian(j, xp, xn):
  res = 1
  xj = xn[j]
  for x in xn:
    if x != xj:
      res *= (xp-x)/(xj-x)
  return res

def interpolate(nodes, x_values):
  xn = np.linspace(1, 2, nodes)
  yn = np.sin(xn)
  y_values = np.zeros(len(x_values))

  for x_index, x in enumerate(x_values):
    for index, y in enumerate(yn):
      y_values[x_index] += y * Lagrangian(index, x, xn)
  
  return y_values

def error(x, function, pmax):
  ytrue = function(x)
  node_max = pmax + 1
  nodes = [i for i in range(2, node_max+1)]
  res = []
  
  for node in nodes:
    res.append(ytrue - interpolate(node, [x])[0])

  return res

# %%
dx = 0.05
x = np.arange(0, 3+dx, dx)
ytrue = np.sin(x)
p1 = interpolate(2, x)
p2 = interpolate(3, x)
p3 = interpolate(4, x)
plt.plot(x, ytrue, label='f(x)')
plt.plot(x, p1, label='$p_{1}(x)$')
plt.plot(x, p2, label='$p_{2}(x)$')
plt.plot(x, p3, label='$p_{3}(x)$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# %%
maxp = 14
p = [f'p{i}(x)' for i in range(1,maxp+1)]
errors = zip(p, error(np.pi/2, np.sin, maxp))
for p, err in errors:
  print(p, err)

# %%
# Task B

def NewtDivDiff(xp, xn, yn):
  n = len(xn)
  a = []
  for i in range(n):
      a.append(yn[i])

  for j in range(1, n):
    for i in range(n-1, j-1, -1):
      a[i] = (a[i]-a[i-1])/(xn[i]-xn[i-j])

  n = len(a) - 1
  temp = a[n] + (xp - xn[n])
  for i in range(n-1, -1, -1 ):
      temp = temp * (xp - xn[i]) + a[i]
  return temp

def NewtInterpolate(nodes, x_values, func):
  y_values = np.zeros(len(x_values))
  xn = np.linspace(1, 2, nodes)
  yn = func(xn)

  for x_index, x in enumerate(x_values):
    y_values[x_index] = NewtDivDiff(x, xn, yn)
  
  return y_values

def Runge(nodes, x_values, func):
  y_values = np.zeros(len(x_values))
  xn = np.linspace(-1, 1, nodes)
  yn = func(xn)

  for x_index, x in enumerate(x_values):
    y_values[x_index] = NewtDivDiff(x, xn, yn)
  
  return y_values

# %%
dx = 0.05
x = np.arange(0, 3+dx, dx)
ytrue = np.sin(x)
p1 = NewtInterpolate(2, x, np.sin)
p2 = NewtInterpolate(3, x, np.sin)
p3 = NewtInterpolate(4, x, np.sin)
plt.plot(x, ytrue, label='f(x)')
plt.plot(x, p1, label='$p_{1}(x)$')
plt.plot(x, p2, label='$p_{2}(x)$')
plt.plot(x, p3, label='$p_{3}(x)$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
# %%
dx = 0.05
x = np.arange(-1, 1+dx, dx)
f = lambda x: 1/(1+25*(x**2))
ytrue = f(x)

plt.plot(x, ytrue, label='f(x)')

for i in range(2, 15):
  p = Runge(i, x, f)
  plt.plot(x, p, label=f'$p_{i-1}(x)$')

plt.ylim(-2, 2)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
# %%
# Task C

f = lambda x: 1/(1+25*(x**2))
df = lambda x: -50*x/(1+25*(x**2))**2

def spline(nodes, a, b, f, df):
  xn = np.linspace(a, b, nodes)
  yn = f(xn)
  dx = xn[1] - xn[0]

  A = np.zeros((nodes, nodes))
  A[0][0] = A[-1][-1] = 1

  D = np.zeros(nodes)
  D[0] = df(a)
  D[-1] = df(b)

  for i in range(1, len(A)-1):
    A[i][i-1] = 1
    A[i][i] = 4 
    A[i][i+1] = 1
    D[i] = 3/dx * (yn[i+1]-yn[i-1])

  V = np.matmul(np.linalg.inv(A), D.T)

  plotSplines(xn, yn, V)

def plotSplines(xn, yn, V):
  dx = 0.05
  DX = xn[1]-xn[0]
  x_values = np.arange(-1, 1+dx, dx)
  y_values = np.zeros(len(x_values))

  pointer = 1
  a = yn[0]
  b = V[0]
  c = 3*(yn[1]-yn[0])/(DX)**2 - (V[1]+2*V[0])/(DX)
  d = -2*(yn[1]-yn[0])/(DX)**3 + (V[1]+V[0])/(DX)**2
  for x_index, x in enumerate(x_values):
    if x > xn[pointer]:
      pointer += 1
      if pointer >= len(xn):
        break
      a = yn[pointer-1]
      b = V[pointer-1]
      c = 3*(yn[pointer]-yn[pointer-1])/(DX)**2 - (V[pointer]+2*V[pointer-1])/(DX)
      d = -2*(yn[pointer]-yn[pointer-1])/(DX)**3 + (V[pointer]+V[pointer-1])/(DX)**2

    diff = x-xn[pointer-1]
    y_values[x_index] = a + b*diff + c*(diff**2) + d*(diff**3)
  
  nodes = len(xn)
  plt.plot(x_values, y_values, label=f'{nodes} nodes')




nodes = [3, 5, 11]
x = np.arange(-1, 1+dx, dx)
ytrue = f(x)

plt.plot(x, ytrue, label='f(x)')

for node in nodes:
  spline(node, -1, 1, f, df)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# %%
