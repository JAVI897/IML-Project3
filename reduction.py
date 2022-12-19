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
from collections import Counter
from sys import getsizeof

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
        print('[INFO] Dimensions data matrix: {}'.format(X.shape))
        print('[INFO] Original shape of data matrix: ', X.shape)
        print('[INFO] Storage in bytes data matrix: ', getsizeof(X))
        start = time.time()

        if self.reduction == 'RNN':
            X_red, Y_red = self.rnn(X, Y)
        elif self.reduction == 'RENN':
            X_red, Y_red = self.renn(X, Y)
        elif self.reduction == 'DROP3':
            X_red, Y_red = self.drop3(X, Y)
        else:
            X_red, Y_red = X, Y

        print('[INFO] Shape of new data matrix: ', X_red.shape)
        print('[INFO] Storage in bytes new data matrix: ', getsizeof(X_red))
        self.time_computation_reduction = time.time() - start

        self.X = X_red
        self.Y = Y_red

        print('[INFO] Dimensions reduced data matrix: {}'.format(self.X.shape))

        if self.W is None:
            self.compute_weights()

    def rnn(self, X, Y):
        pass

    def renn(self, X, Y):
        X_orig, Y_orig = X.copy(), Y.copy()
        self.X = X
        self.Y = Y
        self.compute_weights()
        dM = self.computeDistanceMatrix(X)
        N = X.shape[0]
        knn_indexes = [np.argsort(dM[i, :])[1:self.k+1] for i in range(N)]
        labels_of_neighbours = [Y[indexes].astype(np.int) for indexes in knn_indexes]
        N_c = np.unique(Y).shape[0]
        y_pred = self.vote(labels_of_neighbours, N_c)
        selected_individuals = [m for m in range(N)]
        delete_individuals = 0

        for j in range(N):
            print('[INFO] Analyzing sample: {} Individuals deleted: {}'.format(j, delete_individuals))
            count_dic = Counter([y_pred[l] for l in knn_indexes[j - delete_individuals]])
            if y_pred[j - delete_individuals] != max(count_dic, key=count_dic.get):
                selected_individuals.remove(j)
                delete_individuals += 1
                dM = np.delete(dM, j, axis=0)
                dM = np.delete(dM, j, axis=1)

                X = X_orig[selected_individuals]
                Y = Y_orig[selected_individuals]

                knn_indexes = [np.argsort(dM[i, :])[1:self.k+1] for i in range(X.shape[0])]
                labels_of_neighbours = [Y[indexes].astype(np.int) for indexes in knn_indexes]
                N_c = np.unique(Y).shape[0]
                y_pred = self.vote(labels_of_neighbours, N_c)

        return X_orig[selected_individuals], Y_orig[selected_individuals]

    def drop3(self, X, Y):
        pass