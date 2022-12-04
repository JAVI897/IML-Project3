import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd

#### T-student p-values

def visualize_t_student_matrix(r, savefig_path, N = 10):

    best_N = r.sort_values(by = 'mean_balanced_accuracie', ascending = False).iloc[:N].reset_index()
    best_N['name'] = best_N[['n_neighbors', 'weights', 'metric', 'voting']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    zeros = np.zeros((N, N))
    cols_acc = ['acc_fold_{}'.format(i) for i in range(10)]
    for i1, r1 in best_N.iterrows():
        results_acc_1 = r1[cols_acc].values
        for i2, r2 in best_N.iterrows():
            results_acc_2 = r2[cols_acc].values
            zeros[i1, i2] = round(stats.ttest_ind(results_acc_1, results_acc_2).pvalue, 3)
    matrix_df = pd.DataFrame(zeros,
             columns = best_N.name.values,
               index = best_N.name.values)
    sns.set_style('white', {"grid.color": "0", "grid.linestyle": ":", "grid.color": 'black',
                            'axes.facecolor': '#FFFFFF'})
    fig = plt.figure(figsize=(14,7), facecolor=(1, 1, 1))
    sns.heatmap(matrix_df, annot=True, cmap=sns.diverging_palette(12, 120, n=256))
    plt.title('p-values', loc = 'right')
    plt.grid(False)
    plt.savefig(savefig_path + 'p_values_N_{}.png'.format(N), bbox_inches='tight', dpi=300)

#### K vs metric (accuracy, kappa etc.) with confidence intervals
def visualize_results(r, savefig_path, metric_input = 'accuracie', label_x = 'Accuracy', lim_y=None, categorical_distances = None):
    number_k = list(r['n_neighbors'].unique())
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize= (24.5,28) if categorical_distances is None else (24.5,35) )
    plt.subplots_adjust(wspace=0.15,
                        hspace=0.15)

    for i_m, metric_to_plot in enumerate([ ['minkowski_p_1', 'minkowski_p_3', 'minkowski_p_4'], ['euclidean'], ['cosine']]):
        for m in metric_to_plot:
            for i_w, w in enumerate(['uniform', 'info_gain', 'relief']):
                plt.subplot(4, 3, i_w+1+(3*i_m))
                for v in ['majority', 'inverse_distance', 'shepards']:
                    try:
                        plot_mean = [r.loc[(r['weights'] == w) &
                                           (r['metric']  == m)  &
                                           (r['voting'] == v) &
                                           (r['n_neighbors'] == k)]['mean_{}'.format(metric_input)].values[0] for k in number_k ]

                        plot_sd = [r.loc[(r['weights'] == w) &
                                           (r['metric']  == m)  &
                                           (r['voting'] == v) &
                                           (r['n_neighbors'] == k)]['sd_{}'.format(metric_input)].values[0] for k in number_k ]

                        lower_bound = np.array(plot_mean) - 0.15*np.array(plot_mean)*np.array(plot_sd)
                        upper_bound = np.array(plot_mean) + 0.15*np.array(plot_mean)*np.array(plot_sd)

                        plt.plot( number_k, plot_mean, label='{}-{}-{}'.format(m, w, v), linestyle = 'solid', marker = 'o')
                        plt.fill_between( number_k, upper_bound, lower_bound, alpha=0.3)
                    except:
                        pass
                plt.title('Distance: {} -- Feature weight: {}'.format(m.replace('_4', ''), w))
                if lim_y is not None:
                    plt.ylim(lim_y)
                plt.ylabel(label_x if i_w == 0 else '')
                plt.xlabel('Number of neighbors')
                plt.grid(True)
                plt.legend(loc = 'lower left', prop={'size': 13})

    if categorical_distances is not None:
        for l, v in enumerate(['majority', 'inverse_distance', 'shepards']):
            plt.subplot(4, 3, 10+l)
            try:
                for m in categorical_distances:
                    w = 'uniform'
                    plot_mean = [r.loc[(r['weights'] == w) &
                                       (r['metric'] == m) &
                                       (r['voting'] == v) &
                                       (r['n_neighbors'] == k)]['mean_{}'.format(metric_input)].values[0] for k in number_k]

                    plot_sd = [r.loc[(r['weights'] == w) &
                                     (r['metric'] == m) &
                                     (r['voting'] == v) &
                                     (r['n_neighbors'] == k)]['sd_{}'.format(metric_input)].values[0] for k in number_k]

                    lower_bound = np.array(plot_mean) - 0.15 * np.array(plot_mean) * np.array(plot_sd)
                    upper_bound = np.array(plot_mean) + 0.15 * np.array(plot_mean) * np.array(plot_sd)

                    plt.plot(number_k, plot_mean, label='{}'.format(m), linestyle='solid', marker='o')
                    plt.fill_between(number_k, upper_bound, lower_bound, alpha=0.3)
            except:
                pass
            plt.title('Mixed distance - Voting: {}'.format( v ))
            if lim_y is not None:
                plt.ylim(lim_y)
            plt.ylabel(label_x if l == 0 else '')
            plt.xlabel('Number of neighbors')
            plt.grid(True)
            plt.legend(loc='lower left', prop={'size': 13})

    plt.savefig(savefig_path+'{}.png'.format(metric_input), bbox_inches='tight', dpi=300)