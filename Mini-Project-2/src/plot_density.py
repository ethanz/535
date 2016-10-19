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

y = get_y(x, 1)

plt.subplot(231)
plt.plot(x, y)

y = get_y(x, 2)
plt.plot(x, y)

y = get_y(x, 5)
plt.plot(x, y)

y = get_y(x, 10)
plt.plot(x, y)

y = get_y(x, 20)
plt.plot(x, y)

# plot p(r*|d) for d = 1, 2, 5, 10, 20

plt.show()
