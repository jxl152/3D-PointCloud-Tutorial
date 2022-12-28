import numpy as np
import scipy
import pylab
import random, math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.style.use('seaborn')


class GMM(object):

    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        # weights of each Gaussian
        self.pi = np.ones(self.n_clusters) / self.n_clusters

    def multivariate_normal(self, data, mean_vec, cov):
        var = scipy.stats.multivariate_normal(mean=mean_vec, cov=cov)
        return var.pdf(data)

    def fit(self, data):
        # Initialization of n_clusters Gaussians
        sub_datasets = np.array_split(data, self.n_clusters)
        self.mu = np.asarray([np.mean(sub_data, axis=0) for sub_data in sub_datasets])
        self.cov = np.asarray([np.cov(sub_data.T) for sub_data in sub_datasets])
        # resp[n][k] is the probability of sample n to be part of cluster of k
        resp = np.zeros((len(data), self.n_clusters))

        # Iteration of applying the EM algorithm
        for _ in range(self.max_iter):
            # E-step to calculate the r matrix
            for k in range(resp.shape[1]):
                resp[:, k] = self.pi[k] * self.multivariate_normal(data, self.mu[k], self.cov[k])
            resp /= np.sum(resp, axis=1)[:, np.newaxis]

            N = np.sum(resp, axis=0)
            # M-step to update mean, covariance, and pi
            self.mu = (resp.T @ data) / N[:, np.newaxis]

            for k in range(self.n_clusters):
                diff = data - self.mu[k]
                self.cov[k] = ((resp[:, k] * diff.T) @ diff) / N[k]

            self.pi = N / np.sum(N)

    def predict(self, data):
        probas = np.empty((len(data), self.n_clusters))
        for k in range(self.n_clusters):
            probas[:, k] = self.multivariate_normal(data, self.mu[k], self.cov[k])
        result = np.argmax(probas, axis=1)
        return result


# data simulation
def generate_X(true_Mu, true_Var):
    # generate the first cluster of data
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # generate the second cluster
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # generate the third cluster
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # merge the three clusters into one dataset
    X = np.vstack((X1, X2, X3))
    # visualize the data
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


if __name__ == '__main__':
    np.random.seed(42)

    # simulate data
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)

    # verify the correctness
    labels_0, labels_1, labels_2 = np.empty(400), np.empty(600), np.empty(1000)
    labels_0.fill(0)
    labels_1.fill(1)
    labels_2.fill(2)
    labels = np.hstack((labels_0, labels_1, labels_2))
    accuracy = np.sum(cat == labels) / len(labels)
    print(f"accuracy = {accuracy:.3f}")

    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X[np.where(cat == 0), 0], X[np.where(cat == 0), 1], s=5)
    plt.scatter(X[np.where(cat == 1), 0], X[np.where(cat == 1), 1], s=5)
    plt.scatter(X[np.where(cat == 2), 0], X[np.where(cat == 2), 1], s=5)
    plt.show()
