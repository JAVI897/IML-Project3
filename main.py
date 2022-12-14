import argparse
from kfold import kfold, kfold_reduction
import pandas as pd
import os
from visualize import visualize_results, visualize_stat_test_matrix, plot_times

parser = argparse.ArgumentParser()

### run--> python main.py --dataset vote
parser.add_argument("--dataset", type=str, default='adult', choices=['adult', 'hyp', 'pen-based'])
parser.add_argument("--run_experiments", type=str, default='yes', choices=['yes', 'no'])
parser.add_argument("--reduction", type=str, default='no', choices=['yes', 'no'])
parser.add_argument("--visualize_results", type=str, default='yes', choices=['yes', 'no'])
parser.add_argument("--gpu", type=str, default='yes', choices=['yes', 'no'])
con = parser.parse_args()

def configuration():
    config = {
                'dataset':con.dataset,
                'gpu':con.gpu,
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
                                                'n_neighbors': 3,
                                                'weights': 'info_gain',
                                                'metric': 'euclidean',
                                                'voting': 'majority',
                                                'p': 2
                                                },
                                       'adult': {
                                                'n_neighbors': 3,
                                                'weights': 'info_gain',
                                                'metric': 'euclidean',
                                                'voting': 'majority',
                                                'p': 2
                                                },
                                       'pen-based': {
                                                     'n_neighbors': 3,
                                                     'weights': 'info_gain',
                                                     'metric': 'euclidean',
                                                     'voting': 'majority',
                                                     'p': 2
                                                     }
                                       }
            best_hyp = datasets_config_best_hyp[config['dataset']]
            for reduction_alg in ['RNN', 'RENN', 'DROP3']:
                best_hyp['reduction'] = reduction_alg
                kfold_reduction(config, best_hyp)

        else:
            #### GRIDSEARCH
            for weight in ['relief']: ## ['uniform', 'info_gain', 'relief']
                for metric in ['euclidean']: ## ['minkowski', 'euclidean', 'cosine', 'euclidean-hamming', 'cosine-hamming']
                    for vot in ['shepards']: ## ['majority', 'inverse_distance', 'shepards']
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
                visualize_results(r, savefig_path, metric_input = 'balanced_accuracie', label_x='Balanced accuracy', lim_y=[0.4,1], categorical_distances = ['euclidean-hamming', 'cosine-hamming'] )
                visualize_results(r, savefig_path, metric_input='kappa', label_x='Kappa Index', lim_y=[0, 0.8], categorical_distances = ['euclidean-hamming', 'cosine-hamming'] )
                visualize_results(r, savefig_path, metric_input='macro_precision', label_x='Average Precision', lim_y=[0.4, 0.85], categorical_distances = ['euclidean-hamming', 'cosine-hamming'])
                visualize_results(r, savefig_path, metric_input='macro_recall', label_x='Average Recall', lim_y=[0.5, 1], categorical_distances = ['euclidean-hamming', 'cosine-hamming'])
                visualize_stat_test_matrix(r, savefig_path, N = 10)
                visualize_stat_test_matrix(r, savefig_path, N = 10, stat ='wilcoxon')
                plot_times(r, savefig_path)

        if config['dataset'] == 'pen-based':
            # read results
            path_results = './results/results_pen-based.csv'
            r = pd.read_csv(path_results)
            r = r.drop_duplicates(subset=['n_neighbors', 'weights', 'metric', 'voting'])
            if os.path.isfile(path_results):
                savefig_path = './results/{}/'.format(config['dataset'])
                visualize_results(r, savefig_path, metric_input = 'balanced_accuracie', label_x='Balanced accuracy', lim_y=[0.95, 1] )
                visualize_results(r, savefig_path, metric_input='kappa', label_x='Kappa Index', lim_y=[0.95, 1] )
                visualize_results(r, savefig_path, metric_input='macro_precision', label_x='Average Precision', lim_y=[0.95, 1])
                visualize_results(r, savefig_path, metric_input='macro_recall', label_x='Average Recall', lim_y=[0.95, 1])
                visualize_stat_test_matrix(r, savefig_path, N=10)
                visualize_stat_test_matrix(r, savefig_path, N=10, stat='wilcoxon')
                plot_times(r, savefig_path)

        if config['dataset'] == 'adult':
            # read results
            path_results = './results/results_adult.csv'
            r = pd.read_csv(path_results)
            r = r.drop_duplicates(subset=['n_neighbors', 'weights', 'metric', 'voting'])
            if os.path.isfile(path_results):
                savefig_path = './results/{}/'.format(config['dataset'])
                visualize_results(r, savefig_path, metric_input = 'balanced_accuracie', label_x='Balanced accuracy', categorical_distances = ['euclidean-hamming', 'cosine-hamming'] )
                visualize_results(r, savefig_path, metric_input='kappa', label_x='Kappa Index', categorical_distances = ['euclidean-hamming', 'cosine-hamming'] )
                visualize_results(r, savefig_path, metric_input='macro_precision', label_x='Average Precision', categorical_distances = ['euclidean-hamming', 'cosine-hamming'])
                visualize_results(r, savefig_path, metric_input='macro_recall', label_x='Average Recall', categorical_distances = ['euclidean-hamming', 'cosine-hamming'])
                visualize_stat_test_matrix(r, savefig_path, N=10)
                visualize_stat_test_matrix(r, savefig_path, N=10, stat='wilcoxon')
                plot_times(r, savefig_path)

if __name__ == '__main__':
    main()