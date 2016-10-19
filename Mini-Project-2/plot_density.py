import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

x = np.linspace(0, 5, num=100)

# plot p(r|d)
y = np.sin(x)

plt.subplot(231)
plt.plot(x, y)
plt.title("d = 1")

plt.show()
