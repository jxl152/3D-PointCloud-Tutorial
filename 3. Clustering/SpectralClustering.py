import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph


class Spectral_Clustering(object):
    def __init__(self, n_clusters, n_neighbors=10):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)

    def __preprocess_data(self, data):
        # step 1. affinity matrix
        connectivity = kneighbors_graph(data, n_neighbors=self.n_neighbors, include_self=True)
        affinity_matrix = 0.5 * (connectivity + connectivity.T)
        affinity_matrix = affinity_matrix.toarray()

        # step 2. normalized Laplacian matrix
        L = csgraph.laplacian(affinity_matrix, normed=True)

        # step 3. apply svd to get the first (smallest) k eigen_vectors
        eigen_values, eigen_vectors = np.linalg.eig(L)
        sorted_idx = np.argsort(eigen_values)[:self.n_clusters]
        features = eigen_vectors[:, sorted_idx]
        return features

    def fit(self, data):
        features = self.__preprocess_data(data)
        # kmeans
        self.model.fit(features)

    def predict(self, data):
        features = self.__preprocess_data(data)
        # predict
        result = self.model.predict(features)
        return result


if __name__ == "__main__":
    random_state = 21
    X_mn, labels = make_moons(150, noise=.07, random_state=random_state)

    model = Spectral_Clustering(n_clusters=2)
    model.fit(X_mn)
    result = model.predict(X_mn)
    accuracy = np.sum(result == labels) / len(labels)
    print(f"accuracy = {accuracy:.3f}")

    plt.figure(figsize=(10, 8))
    plt.axis([-2, 2.5, -1, 1.5])
    plt.scatter(X_mn[np.where(result == 0), 0], X_mn[np.where(result == 0), 1], s=5)
    plt.scatter(X_mn[np.where(result == 1), 0], X_mn[np.where(result == 1), 1], s=5)
    plt.show()
