import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison

#### T-student p-values

def visualize_stat_test_matrix(r, savefig_path, N = 10, stat = 'ttest'):

    best_N = r.sort_values(by = 'mean_balanced_accuracie', ascending = False).iloc[:N].reset_index()
    best_N['name'] = best_N[['n_neighbors', 'weights', 'metric', 'voting']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    zeros = np.zeros((N, N))
    cols_acc = ['acc_fold_{}'.format(i) for i in range(10)]
    for i1, r1 in best_N.iterrows():
        results_acc_1 = r1[cols_acc].values
        for i2, r2 in best_N.iterrows():
            results_acc_2 = r2[cols_acc].values
            if stat == 'ttest':
                zeros[i1, i2] = round(stats.ttest_ind(results_acc_1, results_acc_2).pvalue, 3)
            elif stat == 'wilcoxon':
                try:
                    zeros[i1, i2] = round(stats.wilcoxon(results_acc_1, results_acc_2).pvalue, 3)
                except:
                    zeros[i1, i2] = 1
    matrix_df = pd.DataFrame(zeros,
             columns = best_N.name.values,
               index = best_N.name.values)
    sns.set_style('white', {"grid.color": "0", "grid.linestyle": ":", "grid.color": 'black',
                            'axes.facecolor': '#FFFFFF'})
    fig = plt.figure(figsize=(14,7), facecolor=(1, 1, 1))
    sns.heatmap(matrix_df, annot=True, cmap=sns.diverging_palette(12, 120, n=256))
    plt.title('p-values', loc = 'right')
    plt.grid(False)
    plt.savefig(savefig_path + 'p_values_N_{}_{}.png'.format(N, stat), bbox_inches='tight', dpi=300)

#### K vs metric (accuracy, kappa etc.) with confidence intervals
def visualize_results(r, savefig_path, metric_input = 'accuracie', label_x = 'Accuracy', lim_y=None, categorical_distances = None, log = False, fill=True, legend_loc='lower left'):
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize= (24.5,35) if categorical_distances is None else (24.5,35) )
    plt.subplots_adjust(wspace=0.15,
                        hspace=0.15)

    for i_m, metric_to_plot in enumerate([ ['minkowski_p_1', 'minkowski_p_3', 'minkowski_p_4'], ['euclidean'], ['cosine']]):
        for m in metric_to_plot:
            for i_w, w in enumerate(['uniform', 'info_gain', 'relief']):
                plt.subplot(4, 3, i_w+1+(3*i_m))
                for v in ['majority', 'inverse_distance', 'shepards']:
                    try:

                        number_k = np.unique(r.loc[  (r['weights'] == w) &
                                           (r['metric']  == m)  &
                                           (r['voting'] == v)]['n_neighbors'].values)

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

                        plt.plot( number_k, plot_mean, label='{}-{}'.format(m, v), linestyle = 'solid', marker = 'o')
                        if fill:
                            plt.fill_between( number_k, upper_bound, lower_bound, alpha=0.3)
                        if log:
                            plt.yscale('log')
                    except:
                        pass
                plt.title('Distance: {} -- Feature weight: {}'.format(m.replace('_4', ''), w))
                if lim_y is not None:
                    plt.ylim(lim_y)
                plt.ylabel(label_x if i_w == 0 else '')
                plt.xlabel('Number of neighbors')
                plt.grid(True)
                plt.legend(loc = legend_loc, prop={'size': 13})

    if categorical_distances is not None:
        for l, v in enumerate(['majority', 'inverse_distance', 'shepards']):
            plt.subplot(4, 3, 10+l)
            try:
                for m in categorical_distances:
                    w = 'uniform'

                    number_k = np.unique(r.loc[(r['weights'] == w) &
                                               (r['metric'] == m) &
                                               (r['voting'] == v)]['n_neighbors'].values)

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
                    if fill:
                        plt.fill_between(number_k, upper_bound, lower_bound, alpha=0.3)
                    if log:
                        plt.yscale('log')
            except:
                pass
            plt.title('Mixed distance - Voting: {}'.format( v ))
            if lim_y is not None:
                plt.ylim(lim_y)
            plt.ylabel(label_x if l == 0 else '')
            plt.xlabel('Number of neighbors')
            plt.grid(True)
            plt.legend(loc=legend_loc, prop={'size': 13})

    plt.savefig(savefig_path+'{}.png'.format(metric_input), bbox_inches='tight', dpi=300)


def plot_times(r, savefig_path):
    r['metric_weights'] = r['metric'] + '-' + r['weights']
    fig = plt.figure(figsize=(15, 7.5))
    plt.subplots_adjust(wspace=0.05,
                        hspace=0.15)
    for i, voting in enumerate(['majority', 'inverse_distance', 'shepards']):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(data=r.loc[r['voting'] == voting], x="mean_exec_time", y="metric_weights")
        plt.xscale('log')
        plt.xlabel('Average time execution')
        plt.title('Voting = {}'.format(voting))
        if i + 1 > 1:
            plt.yticks([])
        plt.ylabel('')
    plt.savefig(savefig_path + 'time_plot_metrics.png', bbox_inches='tight', dpi=300)

def plot_precision_kappa_balanced_acc(r, savefig_path, ylim = None):
    best = r.sort_values(by='mean_balanced_accuracie', ascending=False).iloc[:1].reset_index()
    r['name'] = r[['n_neighbors', 'weights', 'metric', 'voting']].apply(lambda row: '_'.join(row.values.astype(str)),
                                                                        axis=1)
    best['name'] = best[['n_neighbors', 'weights', 'metric', 'voting']].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)

    p_value_results = []
    cols_acc = ['acc_fold_{}'.format(i) for i in range(10)]
    results_acc_best = best[cols_acc].values[0]
    for i1, r1 in r.iterrows():
        results_acc_1 = r1[cols_acc].values
        try:
            p = round(stats.wilcoxon(results_acc_1, results_acc_best).pvalue, 3)
        except:
            p = 1
        p_value_results.append(p)
    r['p_value_best'] = p_value_results

    fig = plt.figure(figsize=(17, 7))

    for i, metric in enumerate(['mean_macro_precision', 'mean_kappa']):
        plt.subplot(1, 2, i + 1)
        sns.scatterplot(data=r.loc[r['p_value_best'] >= 0.05], x='mean_balanced_accuracie',
                        y=metric, hue='voting', s=90,
                        marker='X')

        sns.scatterplot(data=r.loc[r['p_value_best'] < 0.05], x='mean_balanced_accuracie', marker='o',
                        y=metric, s=70, alpha=0.1, hue='voting', legend=False)

        sns.scatterplot(data=best, x='mean_balanced_accuracie',
                        y=metric, marker='X', s=300, color='red')

        plt.title('Balanced accuracy versus {}'.format(metric))
        plt.ylabel(metric)
        plt.xlabel('Balanced accuracy')
        if ylim is not None:
            plt.ylim(ylim)
        plt.grid()
    plt.savefig(savefig_path + 'precision_kappa_balanced_acc.png', bbox_inches='tight', dpi=300)

def tukey_confidence_interval(r, savefig_path, N = 10):
    best_N = r.sort_values(by = 'mean_balanced_accuracie', ascending = False).iloc[:N].reset_index()
    best_N['name'] = best_N[['n_neighbors', 'weights', 'metric', 'voting']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    comparison_df = best_N[['acc_fold_0', 'name']].values
    for n in ['acc_fold_{}'.format(i) for i in range(1, 10)]:
        comparison_df = np.vstack([comparison_df, best_N[[n, 'name']].values])
    comparison_df = pd.DataFrame(comparison_df, columns = ['acc_fold', 'name'])
    comparison_df["acc_fold"] = pd.to_numeric(comparison_df["acc_fold"])

    fold_data = MultiComparison(comparison_df['acc_fold'], comparison_df['name'])
    results = fold_data.tukeyhsd()
    print(results.summary())
    results.plot_simultaneous(comparison_name = best_N.iloc[0]['name'], figsize=(17, 8))
    plt.savefig(savefig_path + 'tukeyHSD.png', bbox_inches='tight', dpi=300)