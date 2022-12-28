import numpy as np
import matplotlib.pyplot as plt


class K_Means(object):
    """
    K-Means clustering.

    Parameters:
        n_clusters: int, default=2
            The number of clusters to form as well as the number of centroids to generate.
        max_iter: int, default=300
            Maximum number of iterations of the k-means algorithm for a single run.
        tolerance: float, default=1e-4
            Relative tolerance with regard to Frobenius norm of the difference in the cluster centers
            of two consecutive iterations to declare convergence.
    """
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centroids = None

    def initialize_centroids(self, data):
        """
        Create cluster centroids using the k-means++ algorithm.

        Parameters
        ----------
        data : numpy array
            The dataset to be used for centroid initialization.

        Returns
        -------
        centroids : numpy array
            Collection of k centroids as a numpy array.
        """
        centroids = [data[np.random.choice(data.shape[0])]]

        for _ in range(1, self.k_):
            dist_sq = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in data])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break

            centroids.append(data[i])

        return np.array(centroids)

    def fit(self, data):
        # 1. randomly select centroid start points
        min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
        self.centroids = self.initialize_centroids(data)

        # 2. iterate to adjust centroids until convergence or reaching max_iter
        for _ in range(self.max_iter_):
            # assign each point to its nearest centroid
            assigned_points = [[] for _ in range(self.k_)]
            for point in data:
                dists = np.linalg.norm(point - self.centroids, axis=1)
                centroid_idx = np.argmin(dists)
                assigned_points[centroid_idx].append(point)

            # update prev_centroids and centroids
            for i, centroid in enumerate(self.centroids):
                # a centroid has no points
                if np.isnan(centroid).any():
                    self.centroids[i] = np.random.uniform(low=min_, high=max_)
                else:
                    self.centroids[i] = np.mean(assigned_points[i], axis=0)
            prev_centroids = self.centroids

            # check convergence
            centroids_dist = np.linalg.norm(prev_centroids - self.centroids, axis=1)
            if np.all(centroids_dist < self.tolerance_):
                return

    def predict(self, p_datas):
        result = []
        for point in p_datas:
            dists = np.linalg.norm(point - self.centroids, axis=1)
            centroid_idx = np.argmin(dists)
            result.append(centroid_idx)
        return result


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)
    np.set_printoptions(precision=3)
    print(f"centroids are: {k_means.centroids}")

    cat = k_means.predict(x)
    print(cat)

    color = ["blue", "red"]
    for i, point in enumerate(x):
        plt.scatter(point[0], point[1], c=color[cat[i]])
    for centroid in k_means.centroids:
        plt.scatter(centroid[0], centroid[1], marker="x", color="black")
    plt.show()

    # prev_centroids = np.asarray([[0, 1], [2, 3]])
    # centroids = np.asarray([[1, 0], [4, 5]])
    # centroids_diff = prev_centroids - centroids
    # centroids_dist = np.linalg.norm(centroids_diff, axis=1)
    # centroids_stop = centroids_dist < 2
    # stop = np.all(centroids_stop)

    # data = np.asarray([[0, 0], [1, 2], [2, 1], [5, 5], [6, 7]])
    # centroids = np.asarray([[1, 0], [4, 5]])
    # for point in data:
    #     dists = np.linalg.norm(point - centroids, axis=1)
    #     centroid_idx = np.argmin(dists)
    # pass