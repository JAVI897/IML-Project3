import torch
import os
import pandas as pd

def save_results(config, knn_config, kfold_results, cols_to_add):
    path = './results/results_{}.csv'.format(config['dataset'])
    cols = ['acc_fold_{}'.format(i) for i in range(10)] + ['time_fold_{}'.format(i) for i in range(10)] + cols_to_add

    df_aux = pd.DataFrame([kfold_results], columns=cols)
    df_aux = df_aux.round(4)
    df_aux['n_neighbors'] = knn_config['n_neighbors']
    df_aux['weights'] = knn_config['weights']
    df_aux['metric'] = '{}'.format(knn_config['metric']) if knn_config['metric'] != 'minkowski' else '{}_p_{}'.format(knn_config['metric'], knn_config['p'])
    df_aux['voting'] = knn_config['voting']

    if os.path.isfile(path):
        df = pd.read_csv(path)
        df_both = pd.concat([df, df_aux], ignore_index=True, sort=False)
        df_both = df_both.drop_duplicates(subset = ['n_neighbors', 'weights', 'metric', 'voting' ])
        df_both.to_csv(path, index=False)
    else:
        df_aux.to_csv(path, index=False)

def euclidean_matrix(X_new, X, W):
    """
    :param X_new:
    :param X:
    :param W:
    :return: euclidean matrix
    """
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] Computations using: {}'.format('GPU' if torch.cuda.is_available() else 'cpu'))
    X_new = torch.from_numpy(X_new).to(device)
    X = torch.from_numpy(X).to(device)
    W = torch.from_numpy(W).to(device)

    d = X_new.unsqueeze(1) - X.unsqueeze(0)
    d = torch.sqrt(torch.sum( W * (d * d), -1))
    print('[INFO] Distance computed!')
    return d.cpu().detach().numpy()

def euclidean_matrix2(X_new, X, W):
    """
    :param X_new:
    :param X:
    :param W:
    :return: Weighted euclidean distance
    A more efficient way of computing euclidean distance
    Adapted to weighted euclidean distance from:
    https://nenadmarkus.com/p/all-pairs-euclidean/
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print('[INFO] Computations using: {}'.format('GPU' if torch.cuda.is_available() else 'cpu'))
    X_new = torch.from_numpy(X_new).to(device)
    X = torch.from_numpy(X).to(device)
    W = torch.from_numpy(W).to(device)

    sqrX_new = torch.sum( W*torch.pow(X_new, 2), 1, keepdim=True).expand(X_new.shape[0], X.shape[0])
    sqrX = torch.sum( W*torch.pow(X, 2), 1, keepdim=True).expand(X.shape[0], X_new.shape[0]).t()

    d = torch.sqrt( sqrX_new - 2*torch.mm(W*X_new, torch.transpose(X, 0, 1)) + sqrX )
    print('[INFO] Distance computed!')
    return d.cpu().detach().numpy()

def minkowski_matrix(X_new, X, W, p):
    """
    :param X_new:
    :param X:
    :param W:
    :param p:
    :return: minkowski distance
    """
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] Computations using: {}'.format('GPU' if torch.cuda.is_available() else 'cpu'))
    X_new = torch.from_numpy(X_new).to(device)
    X = torch.from_numpy(X).to(device)
    W = torch.from_numpy(W).to(device)

    d = X_new.unsqueeze(1) - X.unsqueeze(0)
    d = torch.sum( W * (torch.abs(d)**p), -1)**(1/p)
    print('[INFO] Distance computed!')
    return d.cpu().detach().numpy()


def cosine_matrix(X_new, X, W):
    """
    :param X_new:
    :param X:
    :param W:
    :return: cosine distance
    """
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] Computations using: {}'.format('GPU' if torch.cuda.is_available() else 'cpu'))
    X_new = torch.from_numpy(X_new).to(device)
    X = torch.from_numpy(X).to(device)
    W = torch.from_numpy(W).to(device)

    sqrt_W = torch.sqrt(W)
    X_new = X_new * sqrt_W
    X = X * sqrt_W
    d = torch.matmul( X_new, torch.transpose(X, 0, 1) ) / torch.linalg.norm(X, dim = 1) / torch.linalg.norm(X_new, dim = 1, keepdim = True)
    print('[INFO] Distance computed!')
    return 1 - d.cpu().detach().numpy()