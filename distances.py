import numpy as np


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
    for i in range(len(u)):
        m = m + (u_v[i] ** p) * w[i]

    m = m ** (1 / p)
    return m


def minkowski_matrix(u, v, w, p):
    if p <= 0:
        raise ValueError("p must be greater than 0")

    if validate_dimensions(u, v) == 0:
        if validate_weights(u, w) == 0:

            D = np.zeros((len(u[0]), len(v)))
            for j in range(len(u[0])):
                for i in range(len(u)):
                    D[i][j] = minkowski(u[i], v[j], w[i], p)
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


