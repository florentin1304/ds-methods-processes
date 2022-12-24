import pandas as pd
import numpy as np
import math

def eucDist(x1,x2):
    return math.sqrt(sum([(v1-v2)**2 for (v1,v2) in list(zip(x1,x2))]))

def silhouette_samples(X, labels):
    """Evaluate the silhouette for each point and return them as a list.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : array, shape = N
    """

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    n_points = X.shape[0]

    unique, counts = np.unique(labels, return_counts=True)
    n_labels = len(unique)

    n_points_per_label = {}
    for u,c in list(zip(unique,counts)):
        n_points_per_label[u] = c

    s = np.empty((n_points,))
    for i, point in enumerate(X):
        distances_by_cluster = np.zeros((n_labels,))
        my_cluster = labels[i]
        for j, point2 in enumerate(X):
            if i == j: continue

            dist = eucDist(point,point2)
            distances_by_cluster[labels[j]] += dist

        a = distances_by_cluster[my_cluster] / (n_points_per_label[my_cluster] - 1)
        average_distances_by_cluster = np.empty((n_labels,))

        for l in range(n_labels):
            average_distances_by_cluster[l] = distances_by_cluster[l] / n_points_per_label[l]

        average_distances_by_cluster[my_cluster] = float('inf')

        b = min(average_distances_by_cluster)

        s[i] = (b-a)/max(b,a)

        if i % 1000 == 0:
            print(i)

    return s


def silhouette_score(X, labels):
    """Evaluate the silhouette for each point and return the mean.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : float
    """
    array_silh = silhouette_samples(X, labels)
    return array_silh.mean()
