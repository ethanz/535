import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import math
import networkx as nx
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.covariance import GraphLasso

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

plt.figure(figsize=(12, 12))
plt.subplot(221)
nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, node_color='red')
#print(precision, covariance, a, b)
plt.axis([0,1,0,1])
plt.axis('on')
plt.title('Adjacency graph')

#plt.show()

plt.matshow(precision, cmap=plt.cm.gray)
plt.title("True precision matrix")
#plt.show()
plt.matshow(covariance, cmap=plt.cm.gray)
plt.title("True covariance matrix")
#plt.show()

# Create Dataset
mean = [1.5]*20
data = np.random.multivariate_normal(mean, covariance, 1000)
cov = np.cov(data)
plt.matshow(cov, cmap=plt.cm.gray)
plt.title("Sample covariance matrix")
plt.matshow(np.asmatrix(cov).I, cmap=plt.cm.gray)
plt.title("Sample precision matrix")
#plt.show()

# PCA
pca = PCA(n_components=0.95)
X = pca.fit(data[:750])
plt.matshow(X.components_, cmap=plt.cm.gray)
plt.title("Principal directions of the data")
plt.matshow(X.get_covariance(), cmap=plt.cm.gray)
plt.title("Estimated covariance matrix")
plt.matshow(X.get_precision(), cmap=plt.cm.gray)
plt.title("Inverse covariance matrix")
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

# GraphLasso

# PPCA
plt.show()
