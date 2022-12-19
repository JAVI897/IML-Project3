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
        # DROP3
        ##fit and predict
        self.X = X
        self.Y = Y
        self.compute_weights()
        dM = self.computeDistanceMatrix(X)
        knn_indexes = [np.argsort(dM[i, :])[1:self.k + 1] for i in range(X.shape[0])]
        labels_of_neighbours = [Y[indexes].astype(np.int) for indexes in knn_indexes]
        N_c = np.unique(Y).shape[0]
        y_pred = self.vote(labels_of_neighbours, N_c)

        ## Remove instances from original data
        remove_intances = [i for i, y_real in enumerate(Y) if y_pred[i] != y_real]
        if len(remove_intances) > 0:
            print('[INFO] Individuals deleted: {}'.format(len(remove_intances)))
            X_removed = np.delete(X, remove_intances, axis=0)
            Y_removed = np.delete(Y, remove_intances)

        # DROP1
        ##fit K+1
        self.X = X_removed
        self.Y = Y_removed
        self.compute_weights()
        dM = self.computeDistanceMatrix(X_removed)

        ##obtain neigbous
        knn_indexes = [np.argsort(dM[i, :])[1:self.k + 1] for i in range(X_removed.shape[0])]
        labels_of_neighbours = [Y_removed[indexes].astype(np.int) for indexes in knn_indexes]
        N_c = np.unique(Y_removed).shape[0]
        y_pred = self.vote(labels_of_neighbours, N_c)

        ##obtain associates
        associates = [[]] * len(X)
        for p_idx in range(X_removed.shape[0]):
            for a_inx in knn_indexes[p_idx]:
                associates[a_inx].append(p_idx)

        # DROP2 :
        ## order of X_removed for removing the instances
        min_dist, instances, classes = [], [], []

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
            instances.append(p)
            classes.append(y_real)

        df = pd.DataFrame({'x': instances, 'dist': min_dist, 'y': classes})
        df = df.sort_values(['dist'], ascending=False)
        X_ordered = X_removed[np.array(list(df['x']))]
        y_ordered = Y_removed[ np.array(list(df['y']))]

        # DROP1
        S = [i for i in range(X_ordered.shape[0])]


        p = 0
        while p < len(S):

            ## Num. of associates of p classified correctly with p as a neighbour
            ###fit and predict

            X_with = X_ordered[S,:]
            y_with = y_ordered[S]
            self.X = X_with
            self.Y = y_with
            self.compute_weights()
            dM = self.computeDistanceMatrix(X_with)
            knn_indexes = [np.argsort(dM[i, :])[1:self.k + 1] for i in range(X_with.shape[0])]
            labels_of_neighbours = [y_with[indexes].astype(np.int) for indexes in knn_indexes]
            N_c = np.unique(y_with).shape[0]
            y_pred = self.vote(labels_of_neighbours, N_c)

            print('[INFO] Instance analysed: {}'.format(p))

            ### with
            try:
                d_with = sum(map(lambda x: y_with[x] == y_pred[x], associates[p]))
            except:
                print('[INFO] Exception within')
                p += 1
                continue

            ## Num. of associates of p classified correctly without p as a neighbour.
            S_without = S.copy()
            del S_without[p]
            ###fit and predict
            X_without = X_ordered[S_without,:]
            y_without = y_ordered[S_without]
            self.X = X_without
            self.Y = y_without
            self.compute_weights()
            dM = self.computeDistanceMatrix(X_without)
            knn_indexes_without = [np.argsort(dM[i, :])[1:self.k + 1] for i in range(X_without.shape[0])]
            labels_of_neighbours = [y_without[indexes].astype(np.int) for indexes in knn_indexes_without]
            N_c = np.unique(y_without).shape[0]
            y_pred = self.vote(labels_of_neighbours, N_c)

            ###without
            try:
                d_without = sum(map(lambda x: y_without[x] == y_pred[x], associates[p]))
            except:
                p += 1
                continue


            if d_without >= d_with:
                print('[INFO] Instance deleted: {}'.format(p))
                S = S_without

                ###obtain associates
                associates = [[]] * len(X)
                for p_idx in range(X_without.shape[0]):
                    for a_inx in knn_indexes_without[p_idx]:
                        associates[a_inx].append(p_idx)

            p += 1

        return X[S], y[S]
