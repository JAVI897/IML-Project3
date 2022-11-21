import numpy as np
from sklearn.feature_selection import mutual_info_classif
import sklearn_relief as relief

class KNN:
    def __init__(self, n_neighbors=5,
                 weights='uniform',
                 metric='minkowski',
                 voting='majority',
                 p=1,
                 ):

        self.k = n_neighbors
        self.voting = voting
        self.weights = weights
        self.metric = metric
        self.p = p

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.compute_weights() # crea self.W (los pesos)

    def predict(self, X_new):
        N = X_new.shape[0]
        distances = self.computeDistanceMatrix(X_new, self.X, self.W)
        # compute k nearest neighbours for each instance
        knn_indexes = [np.argsort(distances[i,:])[:self.k] for i in range(N)]
        labels_of_neighbours = [self.Y[indexes].astype(np.int) for indexes in KNN_indexes]
        N_c = len(self.Y.unique())
        y_pred = self.vote(labels_of_neighbours, N_c, distances)
        return y_pred

    def fit_predict(self, X, Y):
        return self.fit(X, Y).predict(X)

    def compute_weights(self):
        # TO DO haciendo uso del self.X y del self.Y
        # weights va a ser un vector de dimensiones self.X.shape[1] --> numero de columnas (o variables)

        if self.weights == 'equal':
            self.W = np.ones((1, self.X.shape[1]))

        elif self.weights == 'info_gain':
            self.W = mutual_info_classif(self.X, self.Y)

        elif self.weights == 'relief':
            r = relief()
            r.fit_transform(self.X, self.Y)
            self.W = r.w_

    def computeDistanceMatrix(self, X_new, X, W):
        # TO DO
        # las funciones de distancia q implementemos se pueden poner en otro archivo py
        # e importarlas desde aquí o implementarlas aqui
        #if self.metric == 'euclidean':
            #distances = euclidean_dist(X_new, X, w=W)
            # similar a lo que hace la funcion scipy.spatial.distance.cdist
            # pero implementado de forma manual
            # distances será una matriz de dimensiones lxn donde:
            # - l --> numero de instancias de la matriz X_new
            # - n --> numero de instancias de la matriz X
            # cada elemento de la matriz distances D[i, j]
            # sera la distancia del elemento i al elemento j, siendo
            # i una muestra de la matriz X_new y j una muestra de la matriz X
            # OJO: distancia ponderada: hay que usar la matriz W
        # if self.metric == 'minkowski':
            # usar el self.p
            # distances = minkowski_dist(X_new, X, w=W)
        # if self.metric == 'cosine':
            # distances = cosine_dist(X_new, X, w=W)
        #return distances
        return

    def vote(labels, number_classes, Y, voting, distances=None):
        voting_list = []
        for i_x, nearest_labels in enumerate(labels):
            voting_class_list = []
            for class_y in range(number_classes):
                if voting == 'majority':

                    voting_value_class = sum(1 for i in nearest_labels if i == class_y)

                elif voting == 'inverse_distance':
                    voting_value_class = sum(1 / abs(distances(i_x, i)) for i in nearest_labels if i == class_y)

                elif voting == 'shepards':
                    voting_value_class = sum(exp(- abs(distances(i_x, i))) for i in nearest_labels if i == class_y)
                voting_class_list.append(voting_value_class)

            winners = np.argwhere(voting_class_list == np.max(voting_class_list)).flatten()
            if len(winners) > 1:  # tie
                total_class = []
                for winner_class in winners:
                    total_class.append(sum(1 for i in Y if i == winner_class))

                voting_list.append(winners[np.argmax(total_class).flatten()][0])
            else:  # non tie
                voting_list.append(winners[0])
        return voting_list