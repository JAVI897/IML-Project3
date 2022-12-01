import argparse
import time
from datasets import preprocess_adult, preprocess_hypothyroid, preprocess_pen_based
from knn import KNN
from sklearn.metrics import accuracy_score
import numpy as np
import os
from utils import save_results

parser = argparse.ArgumentParser()

### run--> python main.py --dataset vote
parser.add_argument("--dataset", type=str, default='adult', choices=['adult', 'hyp', 'pen-based'])
parser.add_argument("--gpu", type=str, default='yes', choices=['yes', 'no'])
con = parser.parse_args()

def configuration():
    config = {
                'dataset':con.dataset,
                'gpu':con.gpu
             }
    return config

def kfold(config, knn_config: dict, use_precomputed = False):
    accuracies = []
    exec_time = []

    select_function_name = {'adult': ('adult', preprocess_adult),
                            'hyp': ('hypothyroid', preprocess_hypothyroid ),
                            'pen-based': ('pen-based', preprocess_pen_based)}

    for i in range(10):
        dataset_name, preprocess_func = select_function_name[config['dataset']]
        start = time.time()
        X_train, X_test, Y_train, Y_test = preprocess_func('./10_folds/{}/{}.fold.00000{}.train.arff'.format(dataset_name, dataset_name, i ),
                                                           './10_folds/{}/{}.fold.00000{}.test.arff'.format(dataset_name, dataset_name, i ))

        precomputed_distance = None
        distance_name_file = 'distance_{}_p_{}_weight_{}.npy'.format(knn_config['metric'], knn_config['p'], knn_config['weights'])
        path_distance = './precomputed/{}/fold_{}/'.format(dataset_name,i) + distance_name_file

        if use_precomputed:
            if os.path.isfile( path_distance ):
                with open(path_distance, 'rb') as f:
                    precomputed_distance = np.load(f)
                    print('[INFO] Found precomputed distance: {}'.format(distance_name_file))

        precomputed_weights = None
        weight_name_file = 'weight_{}.npy'.format(knn_config['weights'])
        path_weight = './precomputed/{}/'.format(dataset_name) + weight_name_file

        if use_precomputed:
            if os.path.isfile(path_weight):
                with open(path_weight, 'rb') as f:
                    precomputed_weights = np.load(f)
                    print('[INFO] Found precomputed weights: {}'.format(weight_name_file))

        knn_model = KNN(
                     n_neighbors = knn_config['n_neighbors'],
                     weights     = knn_config['weights'],
                     metric      = knn_config['metric'],
                     voting      = knn_config['voting'],
                     p           = knn_config['p'],
                     distances_precomputed = precomputed_distance,
                     weights_precomputed = precomputed_weights,
                     metric_gpu = True if config['gpu'] == 'yes' else False
                     )
        knn_model.fit(X_train.values, Y_train.values)
        y_pred = knn_model.predict(X_test.values)

        # Save distance and weight if they have not been saved before
        if os.path.isfile(path_distance) is False:
            with open(path_distance, 'wb') as f:
                np.save(f, knn_model.distances)

        if os.path.isfile(path_weight) is False:
            with open(path_weight, 'wb') as f:
                np.save(f, knn_model.W)

        running_time = time.time() - start
        exec_time.append(running_time)
        accuracies.append(accuracy_score(y_pred, Y_test))

    mean_acc = np.mean(accuracies)
    sd_mean_acc = np.std(accuracies)
    mean_exec_time = np.mean(exec_time)
    sd_exec_time = np.std(exec_time)
    kfold_results = accuracies + exec_time + [mean_acc, sd_mean_acc, mean_exec_time, sd_exec_time]
    save_results(config, knn_config, kfold_results)
    return mean_acc, sd_mean_acc, mean_exec_time, sd_exec_time

def main():
    config = configuration()

    for weight in ['uniform', 'info_gain', 'relief']:
        for metric in ['minkowski', 'euclidean', 'cosine']:
            for vot in ['majority', 'inverse_distance', 'shepards']:
                for k in range(1, 30, 5):
                    knn_config = {'n_neighbors': k,
                                  'weights': weight,
                                  'metric': metric,
                                  'voting': vot,
                                  'p': 2,
                                  }
                    if metric == 'minkowski':
                        for p in [1, 3, 4]:
                            knn_config['p'] = p
                            mean_acc, sd_mean_acc, mean_exec_time, sd_exec_time = kfold(config, knn_config)
                            print('[INFO] Mean accuracy: {}'.format(mean_acc))
                    else:
                        mean_acc, sd_mean_acc, mean_exec_time, sd_exec_time = kfold(config, knn_config)
                        print('[INFO] Mean accuracy: {}'.format(mean_acc))

if __name__ == '__main__':
    main()