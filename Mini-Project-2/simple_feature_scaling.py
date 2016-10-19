import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 6))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples = n_samples, random_state = random_state)

# Original clustering
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Original K-Means Result")

# After simple feature scaling
for sample in X:
    sample[0] *= 5

y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

plt.subplot(212)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Simple Feature Scaling")

plt.show()
