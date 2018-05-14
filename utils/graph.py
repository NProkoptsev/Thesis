import numpy as np
import scipy
from sklearn.metrics.pairwise import pairwise_kernels

def compute_graph(data, cutoff = 1e-4, metric = 'rbf', **kwds):
    d = pairwise_kernels(data, metric = metric, n_jobs = -1, **kwds)
    d[d < cutoff] = 0
    return d

def normalize_adj(adj):
    rowsum = adj.sum(axis = 1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def get_chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k."""
    n = adj.shape[0]
    adj_normalized = normalize_adj(adj)
    laplacian = np.eye(n) - adj_normalized
    largest_eigval, _ = scipy.linalg.eigh(laplacian, eigvals = (n-1, n-1))
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - np.eye(n)

    t_k = list()
    t_k.append(np.eye(n))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        return 2 * scaled_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian).T)#Transpose as we change order in matmul
    return np.array(t_k)

def get_renormalized_adj_matrix(adj):
    n = adj.shape[0]
    adj = adj + np.eye(n)
    rowsum = adj.sum(axis = 1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)