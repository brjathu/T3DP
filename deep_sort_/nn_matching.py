"""
Modified code from https://github.com/nwojke/deep_sort
"""

import numpy as np
import copy

def _pdist(a, b):

    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))

    return r2



def _cosine_distance(a, b, data_is_normalized=False):
    a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
    b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def _nn_euclidean_distance_min(x, y):
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))



def _nn_euclidean_distance_mean(x, y):
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.mean(axis=0))


def _nn_euclidean_distance_knn(x, y):
    distances = _pdist(x, y)
    a = distances
    b = np.argsort(a, axis=0)
    b = b[:5, :]

    c = []
    for i in range(a.shape[1]):
        c.append(a[:, i][b[:, i]])
    c = np.array(c).T
    return np.maximum(0.0, c.mean(axis=0))



def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):

    def __init__(self, metric, matching_threshold, budget=None):


        if metric == "euclidean_mean":
            self._metric = _nn_euclidean_distance_mean
        elif metric == "euclidean_min":
            self._metric = _nn_euclidean_distance_min
        elif metric == "euclidean_knn":
            self._metric = _nn_euclidean_distance_knn
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError( "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
    
        return cost_matrix
