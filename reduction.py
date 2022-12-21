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
        if reduction not in ['RNN', 'RENN', 'DROP3', 'None']:
            raise ValueError("reduction is expected to be named as; 'RNN', 'RENN', 'DROP3' or 'None' got {} instead".format(reduction))
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

    def rnn(self, X, Y): # RNN USING k = 1 (1-NN)
        self.X = X
        self.Y = Y

        n_elements = X.shape[0]

        # initializing the reduced dataset
        X_red = X[0:1, :]  # select the first row of the array X
        Y_red = Y[0:1]  # select the first element of the array y

        for i in range(1, n_elements):  # for each of the elements
            # find the nearest neighbour of example X[i] in the reduced dataset
            dist = np.linalg.norm(X[i] - X_red, axis=1)
            nearest_neighbour = np.argmin(dist)

            # if X[i] is not misclassified by the reduced dataset
            if Y[i] == Y_red[nearest_neighbour]:
                # adding X[i] to the reduced dataset
                X_red = np.concatenate((X_red, X[i:i + 1, :]))
                Y_red = np.concatenate((Y_red, Y[i:i + 1]))
            else:
                # removing the nearest neighbour from the reduced dataset
                X_red = np.delete(X_red, nearest_neighbour, axis=0)
                Y_red = np.delete(Y_red, nearest_neighbour)
                # adding X[i] to the reduced dataset
                X_red = np.concatenate((X_red, X[i:i + 1, :]))
                Y_red = np.concatenate((Y_red, Y[i:i + 1]))

        return X_red, Y_red


    def renn(self, X, Y):
        self.X = X
        self.Y = Y
        self.compute_weights()
        repetitions = 4
        for i in range(repetitions):
            N = self.X.shape[0]
            dM = self.computeDistanceMatrix(self.X)
            knn_indexes = [np.argsort(dM[i, :])[1:self.k+1] for i in range(N)]
            labels_of_neighbours = [self.Y[indexes].astype(np.int) for indexes in knn_indexes]
            N_c = np.unique(self.Y).shape[0]
            y_pred = self.vote(labels_of_neighbours, N_c)
            selected_ind = [i for i in range(N) if y_pred[i] == self.Y[i]]
            self.X, self.Y = self.X[selected_ind], self.Y[selected_ind]

        return self.X, self.Y

    def drop3(self, X, Y):
        # fit and predict original data
        self.X = X
        self.Y = Y
        self.compute_weights()
        dM = self.computeDistanceMatrix(X)
        knn_indexes = [np.argsort(dM[i, :])[1:self.k + 1] for i in range(X.shape[0])]
        labels_of_neighbours = [Y[indexes].astype(np.int) for indexes in knn_indexes]
        y_pred = self.vote(labels_of_neighbours, np.unique(Y).shape[0])

        # Remove instances from original data
        remove_intances = [i for i, y_real in enumerate(Y) if y_pred[i] != y_real]
        if len(remove_intances) > 0:
            print('[INFO] Individuals deleted: {}'.format(len(remove_intances)))
            X_removed = np.delete(X, remove_intances, axis=0)
            Y_removed = np.delete(Y, remove_intances)
        else:
            X_removed, Y_removed = X, Y

        # fit new data without removed instance
        self.X = X_removed
        self.Y = Y_removed
        self.compute_weights()
        dM = self.computeDistanceMatrix(X_removed)
        total_knn_indexes = [np.argsort(dM[i, :]) for i in range(X_removed.shape[0])]

        # obtain k neigbours
        knn_indexes = [knn_indexes[1:self.k + 1] for knn_indexes in total_knn_indexes]
        labels_of_neighbours = [Y_removed[indexes].astype(np.int) for indexes in knn_indexes]
        N_c = np.unique(Y_removed).shape[0]
        y_pred = self.vote(labels_of_neighbours, N_c)

        # obtain associates
        associates = [[] for _ in range(X_removed.shape[0])]
        for p_index in range(X_removed.shape[0]):
            for a_inx in knn_indexes[p_index]:
                associates[a_inx].append(p_index)

        # order of X_removed for removing the instances
        min_dist = []
        for p, y_real in enumerate(Y_removed):
            enemies = []
            for i, y_neighbour in enumerate(labels_of_neighbours[p]):
                indexes = knn_indexes[p]
                if y_neighbour != y_real:
                    distance = dM[indexes[i], p]
                    enemies.append(distance)
            try:
                min_dist.append(min(enemies))
            except:
                min_dist.append(0)
        enemies_order = np.argsort(min_dist)[::-1]

        index_removed = []
        for p_index in enemies_order:

            d_without, d_with = 0, 0
            for a_index in associates[p_index]:
                # with
                if y_pred[a_index] == Y_removed[a_index]:
                    d_with += 1

                # without
                knn_indexes_without = [knn_index for knn_index in total_knn_indexes[a_index] if knn_index != p_index][
                                      1:self.k + 1]
                labels_of_neighbours_without = [Y_removed[knn_indexes_without].astype(np.int)]
                y_pred_without = self.vote(labels_of_neighbours_without, N_c)
                if y_pred_without[0] == Y_removed[a_index]:
                    d_without += 1

            # eq
            if d_without >= d_with:
                print('[INFO] Instance  removed: {}'.format(p_index))
                index_removed.append(p_index)
                for a_index in associates[p_index]:
                    if p_index in associates[a_index]:
                        old_knn_indexes = knn_indexes[a_index]
                        knn_indexes[a_index] = [knn_indexes for knn_indexes in total_knn_indexes[a_index] if
                                                knn_indexes not in index_removed][1:self.k + 1]
                        labels_of_neighbours[a_index] = [Y_removed[a_index].astype(np.int)]
                        y_pred = self.vote(labels_of_neighbours, N_c)

                        new_neigbour_index = list(set(knn_indexes[a_index]) - set(old_knn_indexes))
                        try:
                            for i in new_neigbour_index:
                                associates[new_neigbour_index[i]].append(a_index)
                        except:
                            pass

        if len(index_removed) > 0:
            print('[INFO] Individuals deleted: {}'.format(len(index_removed)))
            X_removed = np.delete(X_removed, index_removed, axis=0)
            Y_removed = np.delete(Y_removed, index_removed)
        return X_removed, Y_removed

