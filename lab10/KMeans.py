import pandas as pd
import numpy as np
import math

def eucDist(x1,x2):
    return math.sqrt(sum([(v1-v2)**2 for (v1,v2) in list(zip(x1,x2))]))

class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
    def fit_predict(self, X):
        """Run the K-means clustering on X.
        :param X: input data points, array, shape = (N,C).
        :return: labels : array, shape = N.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        self.n_points = X.shape[0]
        self.n_dimensions = X.shape[1]

        initial_centroids_index = np.random.randint(low=0, high=self.n_points, size=self.n_clusters)
        self.centroids = X[initial_centroids_index, :]
        self.labels = np.empty((self.n_points, ), dtype=int)

        for iter in range(self.max_iter):

            # Per tutti i punti cerco il centroid piu vicino
            for p in range(self.n_points):

                # Per tutti i centroid calcolo quello piu vicino
                min_dist = float('inf')
                nearest_centroid = -1

                for c in range(self.n_clusters):
                    dist = eucDist(X[p, :],  self.centroids[c, :])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_centroid = c

                self.labels[p] = int(nearest_centroid)

            # Aggiorno i centroids
            num_points = np.zeros( (self.n_clusters, ) )
            new_centroids = np.zeros(self.centroids.shape)
            for i in range(self.n_points):
                centroid_index = self.labels[i]

                num_points[centroid_index] += 1
                new_centroids[centroid_index, :] += X[i, :]

            for i in range(self.n_clusters):
                self.centroids[i] = new_centroids[i] / num_points[i]


        return self.labels







        pass

