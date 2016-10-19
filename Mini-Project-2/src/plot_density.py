import math
import numpy as np
import matplotlib.pyplot as plt

def p_func (r, d):
    surface = sphere(d)
    denominator = (2*math.pi)**(d/2)
    const = math.e**(-r*r/2)
    return surface*r**(d-1)*const/denominator

def sphere (d):
    if d%2 == 1:
        return 2**((d+1)/2)*math.pi**((d-1)/2)/doublefactorial(d-2)
    else:
        return 2*math.pi**(d/2)/math.factorial(d/2-1)

def doublefactorial (n):
    if n <= 0:
        return 1
    else:
        return n*doublefactorial(n-2)

def get_y (x, d):
    tmp = []
    for i in x:
        tmp.append(p_func(i, d))
    return np.array(tmp)

plt.figure(figsize=(15, 6))

x = np.linspace(0, 5, num=100)

# plot p(r|d) for d = 1, 2, 5, 10, 20

points = [1, 2, 5, 10, 20]
plt.subplot(231)
for p in points:
    y = get_y(x, p)
    plt.plot(x,y)

# plot p(r*|d) for d = 1, 2, 5, 10, 20
x = np.array(points)
tmp = []
for i in x:
    tmp.append(p_func(math.sqrt(i-1), i))
y = np.array(tmp)
print (y)
plt.subplot(232)
plt.plot(x, y)

plt.show()
