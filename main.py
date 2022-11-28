# IMPLEMENTAR LO QUE SE LE PASA COMO ARGUMENTO
# dataset, weights, metric, voting, p, etc.
import argparse
from distutils.command.config import config
from datetime import datetime

from django import conf
from datasets import preprocess_adult, preprocess_hypothyroid
from knn import KNN
from sklearn.metrics import accuracy_score
import numpy as np

parser = argparse.ArgumentParser()

### run--> python main.py --dataset vote
parser.add_argument("--dataset", type=str, default='adult', choices=['adult', 'hyp'])
con = parser.parse_args()

def configuration():
    config = {
                'dataset':con.dataset
             }
    return config

def kfold():
    mean_acc = []
    exec_time = []
    if config['dataset'] == 'adult':
        for i in range(10):
            start = datetime.now()
            
            X_train, X_test, Y_train, Y_test  = preprocess_adult('./10_folds/adult/adult.fold.00000' + str(i) + '.train.arff', './10_folds/adult/adult.fold.00000' + str(i) + '.test.arff')
            y_pred = KNN().fit(X_train, Y_train).predict(X_test)

            running_time = (datetime.now() - start).seconds
            exec_time.append(running_time)
            mean_acc.append(accuracy_score(y_pred, Y_test))
        mean_acc = np.mean(mean_acc)
    elif conf['dataset'] == 'hyp':
        for i in range(10):
            start = datetime.now()
                
            X_train, X_test, Y_train, Y_test  = preprocess_hypothyroid('./10_folds/hypothyroid/hypothyroid.fold.00000' + str(i) + '.train.arff', './10_folds/hypothyroid/hypothyroid.fold.00000' + str(i) + '.test.arff')
            y_pred = KNN().fit(X_train, Y_train).predict(X_test)

            running_time = (datetime.now() - start).seconds
            exec_time.append(running_time)
            mean_acc.append(accuracy_score(y_pred, Y_test))
        mean_acc = np.mean(mean_acc)

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