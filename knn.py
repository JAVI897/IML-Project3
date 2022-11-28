import numpy as np
from sklearn.feature_selection import mutual_info_classif
import sklearn_relief as relief
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from utils import *
from math import exp

class KNN:
    def __init__(self, n_neighbors=5,
                 weights='uniform',
                 metric='minkowski',
                 voting='majority',
                 p=1,
                 distances_precomputed = None,
                 weights_precomputed = None,
                 metric_scipy = True
                 ):

        self.k = n_neighbors
        if voting not in ['majority', 'inverse_distance', 'shepards']:
            raise ValueError("voting is expected to be named as; 'majority', 'inverse_distance' or 'shepards', got {} instead".format(metric))
        self.voting = voting
        self.weights = weights
        if metric not in ['minkowski', 'euclidean', 'cosine']:
            raise ValueError("metric is expected to be named as; 'minkowski', 'euclidean' or 'cosine', got {} instead".format(metric))
        if metric == 'minkowski' and p <= 0:
            raise ValueError("p must be greater than 0")
        self.metric = metric
        self.p = p
        self.distances = distances_precomputed
        self.metric_scipy = metric_scipy
        self.W = weights_precomputed

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
        if self.weights == 'uniform':
            self.W = np.ones((self.X.shape[1],))

        elif self.weights == 'info_gain':
            self.W = mutual_info_classif(self.X, self.Y)

        elif self.weights == 'relief':
            self.W = relief.ReliefF().fit(self.X, self.Y).w_

    def computeDistanceMatrix(self, X_new):

        if self.metric == 'euclidean':
            if self.metric_scipy:
                dist_matrix = cdist(X_new, self.X, metric=self.metric, p=2, w=self.W)
            else:
                #dist_matrix = euclidean_matrix(X_new, X, W, p)
                pass

        elif self.metric == 'minkowski':
            if self.metric_scipy:
                dist_matrix = cdist(X_new, self.X, metric=self.metric, p=self.p, w=self.W)
            else:
                #dist_matrix = minkowski_matrix(X_new, X, W, p)
                pass

        elif self.metric == 'cosine':
            if self.metric_scipy:
                dist_matrix = cdist(X_new, self.X, metric=self.metric, w=self.W)
            else:
                # dist_matrix = minkowski_matrix(X_new, X, W, p)
                pass

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