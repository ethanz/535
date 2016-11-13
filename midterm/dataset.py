import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import math
import _ppca
import networkx as nx
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.covariance import GraphLassoCV

def sample(i, j):
    pr = (1/math.sqrt(2*math.pi))*math.exp(-4*((a[i]-b[i])**2 + (a[j]-b[j])**2))
    r = np.random.rand()
    if r < pr:
        return 1
    return 0

# Create Covariance
a = np.random.rand(20)
b = np.random.rand(20)
adj = []
j = 0

for i in range(20):
    adj.append([])
    for j in range(j+1):
        adj[i].append(sample(i, j))
    j += 1

for i in range(19):
    for k in range(i+1,20):
        adj[i].append(adj[k][i])

precision = []
for i in range(20):
    precision.append([])
    for k in range(20):
        if i == k:
            precision[i].append(1)
        elif adj[i][k] == 0:
            precision[i].append(0)
        else:
            precision[i].append(0.245)
precision = np.asmatrix(precision)
covariance = precision.getI()

G = nx.Graph()
for i in range(20):
    G.add_node(i, pos=(a[i], b[i]))
for i in range(20):
    for k in range(20):
        if adj[i][k] == 1:
            G.add_edge(i, k)

fig, ax = plt.subplots(1,1,figsize=(10,10))
nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, node_color='red')
#print(precision, covariance, a, b)
ax.set_axis_on()
ax.set_title('Adjacency graph')

#plt.show()

fig, ax = plt.subplots(1,4,figsize=(20,8))
ax[0].set_axis_off()
ax[1].set_axis_off()
ax[2].set_axis_off()
ax[3].set_axis_off()
ax[1].matshow(precision, cmap=plt.cm.gray)
ax[1].set_title("True precision matrix")
#plt.show()
ax[0].matshow(covariance, cmap=plt.cm.gray)
ax[0].set_title("True covariance matrix")
#plt.show()

# Create Dataset
mean = [1.5]*20
data = np.random.multivariate_normal(mean, covariance, 1000)
cov = np.cov(data, rowvar=False)
ax[2].matshow(cov, cmap=plt.cm.gray)
ax[2].set_title("Sample covariance matrix")
ax[3].matshow(np.asmatrix(cov).I, cmap=plt.cm.gray)
ax[3].set_title("Sample precision matrix")
fig.tight_layout()
#plt.show()

# PCA
pca = PCA(n_components=0.95)
X = pca.fit(data[:750])
fig, ax = plt.subplots(1,3, figsize=(18, 8))
ax[0].set_axis_off()
ax[1].set_axis_off()
ax[2].set_axis_off()
ax[0].matshow(X.components_, cmap=plt.cm.gray)
ax[0].set_title("Principal directions of the data")
ax[1].matshow(X.get_covariance(), cmap=plt.cm.gray)
ax[1].set_title("Estimated covariance matrix from PCA")
ax[2].matshow(X.get_precision(), cmap=plt.cm.gray)
ax[2].set_title("Inverse covariance matrix from PCA")
fig.tight_layout()
train_reconstructed = X.inverse_transform(X.transform(data[:750]))
test_reconstructed = X.inverse_transform(X.transform(data[750:]))
train_error = []
test_error = []
for idx, t in enumerate(train_reconstructed):
    train_error.append(distance.euclidean(t, data[idx])**2)
for idx, t in enumerate(test_reconstructed):
    test_error.append(distance.euclidean(t, data[750+idx])**2)
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.scatter([i for i in range(750)],train_error)
plt.title("Reconstruction error for training set")
plt.subplot(122)
plt.scatter([i for i in range(250)], test_error)
plt.title("Reconstruction error for test set")
# PPCA
ppca = _ppca.PPCA(data[:750])
ppca.fit()
# GraphLasso
gl = GraphLassoCV()
fig, ax = plt.subplots(1,2,figsize=(12,8))
gl.fit(data[:750])
ax[0].set_axis_off()
ax[0].matshow(gl.precision_, cmap=plt.cm.gray)
ax[0].set_title("Adjacency matrix by GraphicLasso")
ax[1].set_axis_off()
ax[1].matshow(gl.covariance_, cmap=plt.cm.gray)
ax[1].set_title("Covariance matrix by GraphicLasso")
plt.tight_layout()

plt.show()
