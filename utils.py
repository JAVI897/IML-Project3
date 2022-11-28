import numpy as np
from scipy.spatial.distance import euclidean

def find_nm(sample, X):
    dist = 100000
    idx = None
    for i, s in enumerate(X):
        tmp = euclidean(sample, s)
        if tmp <= dist:
            dist = tmp
            idx = i
    return idx

def relief(X, Y):
    feature_scores = np.zeros(X.shape[1])
    labels, counts = np.unique(Y, return_counts=True)

    prob = counts / float(len(Y))
    for label in labels:
        # Find the nearest hit for each sample in the subset with the corresponding label
        select = (Y == label)
        tree = KDTree(X[select, :])
        nh = tree.query(X[select, :], k=2, return_distance=False)[:, 1:]
        nh = (nh.T[0]).tolist()

        # Calculate the difference of x with nh
        nh_mat = np.square(np.subtract(
            X[select, :], X[select, :][nh, :])) * -1

        # Find the nearest miss for each sample in the other subset
        nm_mat = np.zeros_like(X[select, :])
        for prob, other_label in zip(prob[labels != label], labels[labels != label]):
            other_select = (Y == other_label)
            nm = []
            for sample in X[select, :]:
                nm.append(find_nm(sample, X[other_select, :]))
            # Calculate the difference of x with nm
            nm_tmp = np.square(np.subtract(
                X[select, :], X[other_select, :][nm, :])) * prob
            nm_mat = np.add(nm_mat, nm_tmp)
        mat = np.add(nh_mat, nm_mat)
        feature_scores += np.sum(mat, axis=0)
    return normalize([feature_scores])

# Validate dimensions of matrices
def validate_dimensions(u, v):
    row_u = len(u)
    clm_u = len(u[0])

    row_v = len(v)
    clm_v = len(v[0])

    if row_u == row_v and clm_u == clm_v:
        return 0
    else:
        return 1


def validate_weights(u, w):
    clm_u = len(u[0])
    clm_w = len(w[0])

    if clm_u == clm_w:
        return 0
    else:
        return 1


############ DISTANCES #############

# Functions for minkowski
def minkowski(u, v, w, p):
    u_v = np.abs(u - v)
    m = 0
    for i in range(u.shape[0]):
        m = m + (u_v[i] ** p) * w[i]
    m = m ** (1 / p)
    return m

def minkowski_matrix(u, v, w, p):
    if p <= 0:
        raise ValueError("p must be greater than 0")
    N = u.shape[0]
    N_prima = v.shape[0]
    D = np.zeros((N, N_prima))
    for i in range(N):
        for j in range(N_prima):
            D[i][j] = minkowski(u[i], v[j], w, p)
    return D

# Function for euclidean
def euclidean_matrix(u, v, w, p):
    euclidean_dist = minkowski_matrix(u, v, w, p=2)
    return euclidean_dist

# Functions for cosine
def cosine(u, v, w=None):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0 or norm_v == 0:
        raise ValueError("At least one of the observations is a zero vector")

    uv = np.dot(u, v)
    cos = uv / (norm_u * norm_v)
    return cos

def cosine_matrix(u, v):
    if validate_dimensions(u, v) == 0:
        D = np.zeros((len(u[0]), len(v)))
        for j in range(len(u[0])):
            for i in range(len(u)):
                D[i][j] = cosine(u[i], v[j])
        return D

# Weighted cosine
def weighted_cosine(u, v, w):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0 or norm_v == 0:
        raise ValueError("At least one of the observations is a zero vector")

    w_cos = 0
    for i in range(len(u)):
        w_cos = w_cos + ((w[i] * u[i] * v[i]) / (norm_u * norm_v))
    return w_cos

def weighted_cosine_matrix(u, v, w):
    if validate_dimensions(u, v) == 0:
        if validate_weights(u, w) == 0:
            D = np.zeros((len(u[0]), len(v)))
            for j in range(len(u[0])):
                for i in range(len(u)):
                    D[i][j] = weighted_cosine(u[i], v[j], w[i])
        return D