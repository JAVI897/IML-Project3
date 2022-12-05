from knn import KNN
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import sklearn_relief as relief
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from utils import *
from math import exp
import time

class reductionKnnAlgorithm(KNN):

    def __init__(self, n_neighbors=5,
                 weights='uniform',
                 metric='minkowski',
                 voting='majority',
                 p=1,
                 distances_precomputed=None,
                 weights_precomputed=None,
                 metric_gpu=True,
                 binary_vbles_mask=None,
                 reduction=None
                 ):
        super(reductionKnnAlgorithm, self).__init__( n_neighbors=n_neighbors,
                                                     weights=weights,
                                                     metric=metric,
                                                     voting=voting,
                                                     p=p,
                                                     distances_precomputed=distances_precomputed,
                                                     weights_precomputed=weights_precomputed,
                                                     metric_gpu=metric_gpu,
                                                     binary_vbles_mask=binary_vbles_mask,
                                                     )
        if reduction not in ['RNN', 'RENN', 'DROP3']:
            raise ValueError("reduction is expected to be named as; 'RNN', 'RENN' or 'DROP3', got {} instead".format(reduction))
        self.reduction = reduction

    def fit(self, X, Y):

        start = time.time()

        if self.reduction == 'RNN':
            X_red, Y_red = self.rnn(X, Y)
        elif self.reduction == 'RENN':
            X_red, Y_red = self.renn(X, Y)
        elif self.reduction == 'DROP3':
            X_red, Y_red = self.drop3(X, Y)
        else:
            X_red, Y_red = X, Y

        self.time_computation_reduction = time.time() - start

        self.X = X_red
        self.Y = Y_red

        if self.W is None:
            self.compute_weights()

    def rnn(self, X, Y):
        pass

    def renn(self, X, Y):
        pass

    def drop3(self, X, Y):
        pass