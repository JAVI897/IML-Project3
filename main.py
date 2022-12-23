import argparse
from kfold import kfold, kfold_reduction
import pandas as pd
import os
from visualize import visualize_results, visualize_stat_test_matrix, plot_times, plot_precision_kappa_balanced_acc, tukey_confidence_interval

parser = argparse.ArgumentParser()

### run--> python main.py --dataset vote
parser.add_argument("--dataset", type=str, default='pen-based', choices=['adult', 'hyp', 'pen-based'])
parser.add_argument("--run_experiments", type=str, default='yes', choices=['yes', 'no'])
parser.add_argument("--reduction", type=str, default='no', choices=['yes', 'no'])
parser.add_argument("--visualize_results", type=str, default='yes', choices=['yes', 'no'])
parser.add_argument("--pytorch", type=str, default='no', choices=['yes', 'no'])
con = parser.parse_args()

def configuration():
    config = {
                'dataset':con.dataset,
                'gpu':con.pytorch,
                'run_experiments': True if con.run_experiments == 'yes' else False,
                'visualize_results': True if con.visualize_results == 'yes' else False,
                'reduction': True if con.reduction == 'yes' else False
             }
    return config

def main():
    config = configuration()

    if config['run_experiments']:
        if config['reduction']:
            ### GRIDSEARCH OVER REDUCTION ALGORITHMS USING
            # BEST RESULTS OF KNN IN THE CORRESPONDING DATASET
            # change best results
            datasets_config_best_hyp = {'hyp': {
                                                'n_neighbors': 50,
                                                'weights': 'info_gain',
                                                'metric': 'euclidean-hamming',
                                                'voting': 'majority',
                                                'p': 2
                                                },
                                       'adult': {
                                                'n_neighbors': 50,
                                                'weights': 'uniform',
                                                'metric': 'euclidean-hamming',
                                                'voting': 'majority',
                                                'p': 2
                                                },
                                       'pen-based': {
                                                     'n_neighbors': 1,
                                                     'weights': 'info_gain',
                                                     'metric': 'minkowski',
                                                     'voting': 'majority',
                                                     'p': 1
                                                     }
                                       }
            best_hyp = datasets_config_best_hyp[config['dataset']]
            for reduction_alg in ["None",'RENN','RNN','DROP3']:
                print('[INFO] Using reduction: {}'.format(reduction_alg))
                best_hyp['reduction'] = reduction_alg
                kfold_reduction(config, best_hyp)

        else:
            #### GRIDSEARCH
            for weight in ['uniform', 'info_gain', 'relief']: ## ['uniform', 'info_gain', 'relief']
                for metric in ['minkowski', 'euclidean', 'cosine', 'euclidean-hamming', 'cosine-hamming']: ## ['minkowski', 'euclidean', 'cosine', 'euclidean-hamming', 'cosine-hamming']
                    for vot in ['majority', 'inverse_distance', 'shepards']: ## ['majority', 'inverse_distance', 'shepards']
                        for k in [1, 3, 5, 7, 15, 25, 50]: #changed from [25,50]
                            knn_config = {'n_neighbors': k,
                                          'weights': weight,
                                          'metric': metric,
                                          'voting': vot,
                                           'p': 2,
                                          }
                            print('[INFO] Running. n_neighbors:{} weights:{} metric:{} voting:{}'.format(k, weight, metric, vot))
                            if metric == 'minkowski':
                                for p in [1, 3, 5, 7, 15, 25, 50]: # [1, 3, 5, 7, 15, 25, 50]
                                    knn_config['p'] = p
                                    kfold(config, knn_config)
                            else:
                                kfold(config, knn_config)

    if config['visualize_results']:
        if config['dataset'] == 'hyp':
            # read results
            path_results = './results/results_hyp.csv'
            r = pd.read_csv(path_results)
            r = r.drop_duplicates(subset=['n_neighbors', 'weights', 'metric', 'voting'])
            if os.path.isfile(path_results):
                savefig_path = './results/{}/'.format(config['dataset'])
                tukey_confidence_interval(r, savefig_path)
                visualize_results(r, savefig_path, metric_input = 'balanced_accuracie', label_x='Balanced accuracy', lim_y=[0.4,1], categorical_distances = ['euclidean-hamming', 'cosine-hamming'] )
                visualize_results(r, savefig_path, metric_input='kappa', label_x='Kappa Index', lim_y=[0, 0.8], categorical_distances = ['euclidean-hamming', 'cosine-hamming'] )
                visualize_results(r, savefig_path, metric_input='macro_precision', label_x='Average Precision', lim_y=[0.4, 0.85], categorical_distances = ['euclidean-hamming', 'cosine-hamming'])
                visualize_results(r, savefig_path, metric_input='macro_recall', label_x='Average Recall', lim_y=[0.5, 1], categorical_distances = ['euclidean-hamming', 'cosine-hamming'])
                visualize_results(r, savefig_path, metric_input='exec_time', label_x='Time execution', lim_y=[0, 85], categorical_distances=['euclidean-hamming', 'cosine-hamming'],
                                  log = False, fill=False, legend_loc = 'center')
                visualize_stat_test_matrix(r, savefig_path, N = 10)
                visualize_stat_test_matrix(r, savefig_path, N = 10, stat ='wilcoxon')
                plot_times(r, savefig_path)
                plot_precision_kappa_balanced_acc(r, savefig_path)

        if config['dataset'] == 'pen-based':
            # read results
            path_results = './results/results_pen-based.csv'
            r = pd.read_csv(path_results)
            r = r.drop_duplicates(subset=['n_neighbors', 'weights', 'metric', 'voting'])
            if os.path.isfile(path_results):
                savefig_path = './results/{}/'.format(config['dataset'])
                tukey_confidence_interval(r, savefig_path)
                visualize_results(r, savefig_path, metric_input = 'balanced_accuracie', label_x='Balanced accuracy', lim_y=[0.95, 1] )
                visualize_results(r, savefig_path, metric_input='kappa', label_x='Kappa Index', lim_y=[0.95, 1] )
                visualize_results(r, savefig_path, metric_input='macro_precision', label_x='Average Precision', lim_y=[0.95, 1])
                visualize_results(r, savefig_path, metric_input='macro_recall', label_x='Average Recall', lim_y=[0.95, 1])
                visualize_results(r, savefig_path, metric_input='exec_time', label_x='Time execution')
                visualize_results(r, savefig_path, metric_input='exec_time', label_x='Time execution', lim_y=[0, 150],
                                  log=False, fill=False, legend_loc='center')
                visualize_stat_test_matrix(r, savefig_path, N=10)
                visualize_stat_test_matrix(r, savefig_path, N=10, stat='wilcoxon')
                plot_times(r, savefig_path)
                plot_precision_kappa_balanced_acc(r, savefig_path, ylim=(0.8, 1))

        if config['dataset'] == 'adult':
            # read results
            path_results = './results/results_adult.csv'
            r = pd.read_csv(path_results)
            r = r.drop_duplicates(subset=['n_neighbors', 'weights', 'metric', 'voting'])
            if os.path.isfile(path_results):
                savefig_path = './results/{}/'.format(config['dataset'])
                tukey_confidence_interval(r, savefig_path)
                visualize_results(r, savefig_path, metric_input = 'balanced_accuracie', label_x='Balanced accuracy', categorical_distances = ['euclidean-hamming', 'cosine-hamming'], lim_y=[0.7, 0.85] )
                visualize_results(r, savefig_path, metric_input='kappa', label_x='Kappa Index', categorical_distances = ['euclidean-hamming', 'cosine-hamming'], lim_y=[0.4, 0.55] )
                visualize_results(r, savefig_path, metric_input='macro_precision', label_x='Average Precision', categorical_distances = ['euclidean-hamming', 'cosine-hamming'], lim_y=[0.7, 0.8])
                visualize_results(r, savefig_path, metric_input='macro_recall', label_x='Average Recall', categorical_distances = ['euclidean-hamming', 'cosine-hamming'], lim_y=[0.7, 0.8])
                visualize_results(r, savefig_path, metric_input='exec_time', label_x='Time execution', lim_y=[0, 800],
                                  categorical_distances=['euclidean-hamming', 'cosine-hamming'],
                                  log=False, fill=False, legend_loc='best')
                visualize_stat_test_matrix(r, savefig_path, N=10)
                visualize_stat_test_matrix(r, savefig_path, N=10, stat='wilcoxon')
                plot_times(r, savefig_path)
                plot_precision_kappa_balanced_acc(r, savefig_path)

if __name__ == '__main__':
    main()