# IMPLEMENTAR LO QUE SE LE PASA COMO ARGUMENTO
# dataset, weights, metric, voting, p, etc.

from datasets import preprocess_adult, preprocess_hypothyroid

def main():
    # cargar el config como siempre
    # leer los kfolds, se podría poner como una función
    # para cada fold; train, test --> preprocesar el training y test a la vez
    # por ejemplo:
    #     X_train, X_test, Y_train, Y_test = preprocess_adult('./10_folds/adult/adult.fold.000000.train.arff',
    #                                                         './10_folds/adult/adult.fold.000000.test.arff')
    pass

if __name__ == '__main__':
    main()