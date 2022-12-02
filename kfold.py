import time
from datasets import preprocess_adult, preprocess_hypothyroid, preprocess_pen_based
from knn import KNN
from sklearn.metrics import accuracy_score, cohen_kappa_score, balanced_accuracy_score, precision_score, recall_score
import numpy as np
import os
import pandas as pd
from utils import save_results

def kfold(config, knn_config: dict, use_precomputed = True):
    accuracies = []
    balanced_accuracies = []
    kappas = []
    macro_precision = []
    macro_recall = []
    exec_time = []

    select_function_name = {'adult': ('adult', preprocess_adult),
                            'hyp': ('hypothyroid', preprocess_hypothyroid ),
                            'pen-based': ('pen-based', preprocess_pen_based)}

    for i in range(10):
        dataset_name, preprocess_func = select_function_name[config['dataset']]
        start = time.time()
        X_train, X_test, Y_train, Y_test = preprocess_func('./10_folds/{}/{}.fold.00000{}.train.arff'.format(dataset_name, dataset_name, i ),
                                                           './10_folds/{}/{}.fold.00000{}.test.arff'.format(dataset_name, dataset_name, i ))

        time_distance_precomputed = None
        precomputed_distance = None
        distance_name_file = 'distance_{}_p_{}_weight_{}.npy'.format(knn_config['metric'], knn_config['p'], knn_config['weights'])
        path_distance = './precomputed/{}/fold_{}/'.format(dataset_name,i) + distance_name_file

        if use_precomputed:
            if os.path.isfile( path_distance ):
                with open(path_distance, 'rb') as f:
                    precomputed_distance = np.load(f)
                    print('[INFO] Found precomputed distance: {}'.format(distance_name_file))
                df_times_distances = pd.read_csv('./precomputed/{}/fold_{}/distances_times.csv'.format(dataset_name,i))
                time_distance_precomputed = df_times_distances.loc[df_times_distances['name'] == distance_name_file]['time'].values[0] # retrieve time to compute distance

        time_weights_precomputed = None
        precomputed_weights = None
        weight_name_file = 'weight_{}.npy'.format(knn_config['weights'])
        path_weight = './precomputed/{}/'.format(dataset_name) + weight_name_file

        if use_precomputed:
            if os.path.isfile(path_weight):
                with open(path_weight, 'rb') as f:
                    precomputed_weights = np.load(f)
                    print('[INFO] Found precomputed weights: {}'.format(weight_name_file))
                df_times_weights = pd.read_csv('./precomputed/{}/weight_times.csv'.format(dataset_name))
                time_weights_precomputed = df_times_weights.loc[df_times_weights['name'] == weight_name_file]['time'].values[0]  # retrieve time to compute distance

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
            times = pd.DataFrame({'name': [distance_name_file], 'time': [knn_model.time_computation_distance]})
            if os.path.isfile('./precomputed/{}/fold_{}/distances_times.csv'.format(dataset_name,i)) is False:
                times.to_csv('./precomputed/{}/fold_{}/distances_times.csv'.format(dataset_name,i))
            else:
                df_times_distances = pd.read_csv('./precomputed/{}/fold_{}/distances_times.csv'.format(dataset_name,i))
                df_both = pd.concat([df_times_distances, times], ignore_index=True, sort=False)
                df_both.to_csv('./precomputed/{}/fold_{}/distances_times.csv'.format(dataset_name,i), index=False)

        if os.path.isfile(path_weight) is False:
            with open(path_weight, 'wb') as f:
                np.save(f, knn_model.W)
            weight_times = pd.DataFrame({'name': [weight_name_file], 'time': [knn_model.time_computation_weights]})
            if os.path.isfile('./precomputed/{}/weight_times.csv'.format(dataset_name)) is False:
                weight_times.to_csv('./precomputed/{}/weight_times.csv'.format(dataset_name))
            else:
                df_times_weights = pd.read_csv('./precomputed/{}/weight_times.csv'.format(dataset_name))
                df_both = pd.concat([df_times_weights, weight_times], ignore_index=True, sort=False)
                df_both.to_csv('./precomputed/{}/weight_times.csv'.format(dataset_name), index=False)

        running_time = time.time() - start
        if use_precomputed and time_distance_precomputed is not None:
            running_time += time_distance_precomputed
        if use_precomputed and time_weights_precomputed is not None:
            running_time += time_weights_precomputed

        exec_time.append(running_time)
        accuracies.append(accuracy_score(y_pred, Y_test))
        kappas.append(cohen_kappa_score(y_pred, Y_test, labels=np.unique(y_pred) ))
        balanced_accuracies.append(balanced_accuracy_score(y_pred, Y_test))
        macro_precision.append(precision_score(y_pred, Y_test, average = 'macro', labels=np.unique(y_pred) ))
        macro_recall.append(recall_score(y_pred, Y_test, average = 'macro', labels=np.unique(y_pred) ))

    cols_to_add = []
    results_mean_sd = []
    for name_metric, metric_folds in [('exec_time', exec_time), ('accuracie', accuracies),
                                      ('kappa', kappas), ('balanced_accuracie', balanced_accuracies),
                                      ('macro_precision', macro_precision), ('macro_recall', macro_recall)]:

        results_mean_sd.append(np.mean(metric_folds))
        cols_to_add.append('mean_{}'.format(name_metric))
        results_mean_sd.append(np.std(metric_folds))
        cols_to_add.append('sd_{}'.format(name_metric))

    kfold_results = accuracies + exec_time + results_mean_sd
    save_results(config, knn_config, kfold_results, cols_to_add)
    return