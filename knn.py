import numpy as np
from sklearn.feature_selection import mutual_info_classif
import sklearn_relief as relief
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from utils import *
from math import exp
import time

class KNN:
    def __init__(self, n_neighbors=5,
                 weights='uniform',
                 metric='minkowski',
                 voting='majority',
                 p=1,
                 distances_precomputed = None,
                 weights_precomputed = None,
                 metric_gpu = True,
                 binary_vbles_mask = None
                 ):

        self.k = n_neighbors
        if voting not in ['majority', 'inverse_distance', 'shepards']:
            raise ValueError("voting is expected to be named as; 'majority', 'inverse_distance' or 'shepards', got {} instead".format(voting))
        self.voting = voting
        if weights not in ['uniform', 'info_gain', 'relief']:
            raise ValueError("weight is expected to be named as; 'uniform', 'info_gain' or 'relief', got {} instead".format(weights))
        self.weights = weights
        if metric not in ['minkowski', 'euclidean', 'cosine', 'euclidean-hamming', 'cosine-hamming']:
            raise ValueError("metric is expected to be named as; 'minkowski', 'euclidean' or 'cosine', got {} instead".format(metric))
        if metric == 'minkowski' and p <= 0:
            raise ValueError("p must be greater than 0")
        self.metric = metric
        self.p = p
        self.distances = distances_precomputed
        self.W = weights_precomputed
        self.metric_gpu = metric_gpu
        self.binary_vbles_mask = binary_vbles_mask

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        if self.W is None:
            self.compute_weights()

    def predict(self, X_new):
        N = X_new.shape[0]
        if self.distances is None:
            self.distances = self.computeDistanceMatrix(X_new)
        # compute k nearest neighbours for each instance
        knn_indexes = [np.argsort(self.distances[i, :])[:self.k] for i in range(N)]
        labels_of_neighbours = [self.Y[indexes].astype(np.int) for indexes in knn_indexes]
        N_c = np.unique(self.Y).shape[0]
        y_pred = self.vote(labels_of_neighbours, N_c)
        return y_pred

    def fit_predict(self, X, Y):
        return self.fit(X, Y).predict(X)

    def compute_weights(self):
        start = time.time()

        if self.weights == 'uniform':
            self.W = np.ones((self.X.shape[1],))

        elif self.weights == 'info_gain':
            self.W = mutual_info_classif(self.X, self.Y)

        elif self.weights == 'relief':
            self.W = relief.ReliefF().fit(self.X, self.Y).w_

        self.time_computation_weights = time.time() - start

    def computeDistanceMatrix(self, X_new):

        start = time.time()

        if self.metric == 'euclidean':
            if self.metric_gpu:
                dist_matrix = euclidean_matrix2(X_new, self.X, self.W)
            else:
                dist_matrix = cdist(X_new, self.X, metric='minkowski', p=2, w=self.W)

        elif self.metric == 'minkowski':
            if self.metric_gpu:
                dist_matrix = minkowski_matrix(X_new, self.X, self.W, self.p)
            else:
                dist_matrix = cdist(X_new, self.X, metric='minkowski', p=self.p, w=self.W)

        elif self.metric == 'cosine':
            if self.metric_gpu:
                dist_matrix = cosine_matrix(X_new, self.X, self.W)
            else:
                dist_matrix = cdist(X_new, self.X, metric='cosine', w=self.W)

        elif self.metric == 'euclidean-hamming':
            binary_vbles = [i for i, v in enumerate(self.binary_vbles_mask) if v == 1]
            numeric_vbles = [i for i, v in enumerate(self.binary_vbles_mask) if v == 0]
            if self.metric_gpu:
                dist_euclidean = euclidean_matrix2(X_new[:,numeric_vbles], self.X[:,numeric_vbles], np.ones((len(numeric_vbles),)) )
            else:
                dist_euclidean = cdist(X_new[:,numeric_vbles], self.X[:,numeric_vbles], metric='minkowski', p=2)
            dist_hamming = cdist(X_new[:,binary_vbles], self.X[:,binary_vbles], metric='hamming', w = np.ones((len(binary_vbles),)) )
            dist_matrix = 1/2 * (dist_euclidean + dist_hamming)

        elif self.metric == 'cosine-hamming':
            binary_vbles = [i for i, v in enumerate(self.binary_vbles_mask) if v == 1]
            numeric_vbles = [i for i, v in enumerate(self.binary_vbles_mask) if v == 0]
            dist_cosine = cosine_matrix(X_new[:,numeric_vbles], self.X[:,numeric_vbles], np.ones((len(numeric_vbles),)) )
            dist_hamming = cdist(X_new[:,binary_vbles], self.X[:,binary_vbles], metric='hamming', w = np.ones((len(binary_vbles),)) )
            dist_matrix = 1/2 * (dist_cosine + dist_hamming)

        self.time_computation_distance = time.time() - start

        return dist_matrix

    def vote(self, labels, number_classes, distances=None):
        voting_list = []

        for i_x, nearest_labels in enumerate(labels):
            voting_class_list = []
            for class_y in range(number_classes):
                if self.voting == 'majority':
                    voting_value_class = sum(1 for i in nearest_labels if i == class_y)
                    voting_class_list.append(voting_value_class)
                elif self.voting == 'inverse_distance':
                    voting_value_class = sum(1 / abs( self.distances[i,i_x] ) for i in nearest_labels if i == class_y)
                    voting_class_list.append(voting_value_class)
                elif self.voting == 'shepards':
                    voting_value_class = sum(exp(- abs( self.distances[i,i_x] )) for i in nearest_labels if i == class_y)
                    voting_class_list.append(voting_value_class)

            winners = np.argwhere(voting_class_list == np.max(voting_class_list)).flatten()

            if len(winners) > 1:  # tie
                total_class = []
                for winner_class in winners:
                    total_class.append(sum(1 for i in self.Y if i == winner_class))
                voting_list.append(winners[np.argmax(total_class).flatten()][0])
            else:  # non tie
                voting_list.append(winners[0])

        return np.array(voting_list)