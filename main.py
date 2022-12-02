import argparse
from kfold import kfold

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

def main():
    config = configuration()
    #### GRIDSEARCH
    for weight in ['uniform']: ## ['uniform', 'info_gain', 'relief']
        for metric in ['euclidean', 'cosine']: ## ['minkowski', 'euclidean', 'cosine']
            for vot in ['majority', 'inverse_distance', 'shepards']: ## ['majority', 'inverse_distance', 'shepards']
                for k in range(1, 30, 5):
                    knn_config = {'n_neighbors': k,
                                  'weights': weight,
                                  'metric': metric,
                                  'voting': vot,
                                  'p': 2,
                                  }
                    print('[INFO] Running. n_neighbors:{} weights:{} metric:{} voting:{}'.format(k, weight, metric, vot))
                    if metric == 'minkowski':
                        for p in [1, 3, 4]:
                            knn_config['p'] = p
                            kfold(config, knn_config)
                    else:
                        kfold(config, knn_config)

if __name__ == '__main__':
    main()