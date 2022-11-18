import numpy as np

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
        policy = self.vote(labels_of_neighbours, N_c, distances)


    def fit_predict(self, X, Y):
        return self.fit(X, Y).predict(X)

    def compute_weights(self):
        # TO DO haciendo uso del self.X y del self.Y
        # weights va a ser un vector de dimensiones self.X.shape[1] --> numero de columnas (o variables)
        if self.weights == 'uniform':
            #self.W = vector
        # elif self.weights == 'otra':
        # no hace falta hacer ningun return

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

    def vote(self, labels, number_classes, distances):
        voting_matrix = [[] for j in range(len(labels))]
        for i_x, nearest_labels in enumerate(labels):
            for class_y in range(number_classes):
                # TO DO
                if self.voting == 'majority':
                    # voting_value_class: suma de votos de la clase en la que se itera (class_y)
                    # para la muestra sobre la que se itera (i_x). Asegurarse de que siempre es un
                    # valor real voting_value_class
                    voting_value_class = sum(1 for i in nearest_labels if i == class_y) # algo así, no se si estará bien
                #elif self.voting == 'inverse_distance':
                # hay que usar la matriz de distancias; distancia entre la sample i_x y el
                # vecino j es distances[i_x, j] dnd j pertenece a nearest_labels
                    #voting_value_class =
                #elif self.voting == 'shepards':
                    #voting_value_class =
                voting_matrix[i_x].append(voting_value_class)

        voting_matrix = np.array(voting_matrix)
        return np.armax(voting_matrix, axis = 1)