import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def euclid(X: np.ndarray, centroids: np.ndarray):
    return np.sqrt(np.sum((X - centroids[:, np.newaxis]) ** 2, axis=2))

def manhattan(X: np.ndarray, centroids: np.ndarray):
    return np.sum(np.abs(X - centroids[:, np.newaxis]), axis=2)

def chebychev(X: np.ndarray, centroids: np.ndarray):
    return np.max(np.abs(X - centroids[:, np.newaxis]), axis=2)

def kmeans(X: np.ndarray, k: int, iter: int):

    centroid_idx = np.random.choice(len(X), k, replace=False)
    centroids = X[centroid_idx]
    
    for _ in range(iter):
        labels = np.argmin(euclid(X, centroids), axis=0)
        new_centroids = np.array([X[labels == cluster].mean(axis=0) for cluster in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids


data = pd.read_csv('./filtrirane_knjige.csv')
data = data[['broj_strana', 'godina_izdavanja', 'tip_poveza', 'povrsina', 'cena']]
data['tip_poveza'] = data['tip_poveza'].apply(lambda x: 1 if x == "Tvrd" else 0)

print(data)

features = {0: 'broj_strana', 1: 'godina_izdavanja', 2: 'tip_poveza', 3: 'povrsina', 4: 'cena'}
f1 = 0
f2 = 2
f3 = 4
k = 3
iterations = 100

data = data.to_numpy()
x = data[:, [f1, f2, f3]]
labels, centroids = kmeans(x, k, iterations)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=labels)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=range(len(centroids)), marker='x', s=100, lw=5)
ax.set_title(f'3D K-means Clustering Results (k={k})')
ax.set_xlabel(features[f1])
ax.set_ylabel(features[f2])
ax.set_zlabel(features[f3])

plt.show()