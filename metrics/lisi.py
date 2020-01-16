import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

# ========================= LISI Implementation 1 ============================
# Where the binary search for perplexity is taken from van der Maaten et al.'s
# python implementation of t-SNE (https://lvdmaaten.github.io/tsne/)
def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P) + np.finfo(np.double).eps
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def compute_simpson_index2(knn_dist, knn_idx, batch_labels, n_batches, perplexity=15, tol=1e-5):
    # First performs a binary search to get P-values in such a way that each
    # conditional Gaussian has the same perplexity.

    # Initialize some variables
    n = knn_dist.shape[0] # n rows (n cells)
    P = np.zeros(knn_dist.shape[1]) # number of neighbors (K)
    simpson = np.zeros(n)
    beta = np.ones(n)
    logU = np.log(perplexity)
    # Loop over all datapoints
    for i in range(n):
        # Compute the Gaussian kernel and entropy for the current precision
        betamin = np.Inf
        betamax = -np.Inf
        H, P = Hbeta(knn_dist[i], beta[i])
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.absolute(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if not np.isfinite(betamax):
                    beta[i] *= 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if not np.isfinite(betamin):
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            # Recompute the values
            H, P = Hbeta(knn_dist[i], beta[i])
            Hdiff = H - logU
            tries += 1
        # end while
        if H == 0:
            simpson[i] = -1
            continue
        
        # Then compute Simpson's Index
        for b in range(n_batches):
            # Get a local neighborhood of cell i
            q = batch_labels[knn_idx[i]] == b # get batch_labels of k nearest neighbors of i, and select those that have batch = b
            if np.sum(q) > 0:
                sumP = np.sum(P[q]) # sumP is "relative abundance" of each batch (each speices)
                simpson[i] += sumP * sumP
    return simpson

def lisi2(X, meta_data, labels_use, perplexity=30, nn_eps=0):
    print('computing LISI score')
    if X.shape[1] > 1000:
        print(X.shape)
        print('Warning: detected high-dimensional (> 100) embedding')
        print('Computing LISI requires nearest-neighbors computations,')
        print('which are expensive in high dimensions.')
        print('Applying PCA to your data first...')
        X = PCA(n_components=100).fit_transform(X)
    N = meta_data.shape[0] # n rows (n cells)
    kdtree = KDTree(X)
    knn_d, knn_idx = kdtree.query(X, k=perplexity*3, eps=nn_eps)
    # Don't count yourself in your neighborhood
    knn_d = knn_d[:, 1:]
    knn_idx = knn_idx[:, 1:]
    lisi = np.empty((N, len(labels_use)))
    lisi[:] = np.nan
    for idx, label in enumerate(labels_use):
        labels = meta_data[label]
        if labels.isnull().values.any():
            print('Cannot compute LISI on missing values')
            continue
        else:
            uniq, labels = np.unique(labels, return_inverse=True) # labels as ints
            n_batches = len(uniq)
            simpson = compute_simpson_index2(knn_d, knn_idx, labels, n_batches, perplexity)
            lisi[:, idx] = 1 / simpson
    lisi_df = pd.DataFrame(lisi, index=meta_data.index, columns=labels_use)
    return lisi_df

# ========================= LISI Implementation 2 ============================
# As direct a translation of Harmony's (Korsunsky et al) implementation in
# R/C++ to python as possible
def compute_hbeta(D, beta, P, idx):
    P = np.exp(-D[:, idx] * beta)
    sumP = np.sum(P)
    if sumP == 0:
        H = 0
        P = D[:, idx] * 0
    else:
        H = np.log(sumP) + beta * np.sum(np.mod(D[:, idx], P)) / sumP
        P /= sumP
    return H, P

'''
knn_dist: shape=(K, N)
knn_idx:  shape=(K, N)

'''
def compute_simpson_index(knn_dist, knn_idx, batch_labels, n_batches, perplexity=15, tol=1e-5):
    n = knn_dist.shape[1] # n_cols
    P = np.zeros(knn_dist.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    hbeta, beta, betamin, betamax, H, Hdiff = [0.] * 6
    tries = 0
    for i in range(n):
        beta = 1
        betamin = np.Inf
        betamax = -np.Inf
        H, P = compute_hbeta(knn_dist, beta, P, i)
        Hdiff = H - logU
        tries = 0
        # First get neighbor probabilities
        while np.absolute(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            H, P = compute_hbeta(knn_dist, beta, P, i)
            Hdiff = H - logU
            tries += 1
        # end while
        if H == 0:
            simpson[i] = -1
            continue
        
        # Then compute Simpson's Index
        for b in range(n_batches):
            # Get a local neighborhood of cell i
            q = batch_labels[knn_idx[:, i]] == b # get batch_labels of k nearest neighbors of i, and select those that have batch = b
            if np.sum(q) > 0:
                sumP = np.sum(P[q]) # sumP is "relative abundance" of each batch (each speices)
                simpson[i] += sumP * sumP
    return simpson

'''Compute Local Inverse Simpson's Index (LISI)

This is a per-cell score (effective number of batches in that cell's neighborhood)
'''
def lisi(X, meta_data, labels_use, perplexity=30, nn_eps=0):
    N = meta_data.shape[0] # n rows (n cells)
    kdtree = KDTree(X)
    knn_d, knn_idx = kdtree.query(X, k=perplexity*3, eps=nn_eps)
    # Don't count yourself in your neighborhood
    knn_d = knn_d[:, 1:]
    knn_idx = knn_idx[:, 1:]
    lisi = np.empty((N, len(labels_use)))
    lisi[:] = np.nan
    for idx, label in enumerate(labels_use):
        labels = meta_data[label]
        if labels.isnull().values.any():
            print('Cannot compute LISI on missing values')
            continue
        else:
            uniq, labels = np.unique(labels, return_inverse=True) # labels as ints
            n_batches = len(uniq)
            simpson = compute_simpson_index(knn_d.T, knn_idx.T, labels, n_batches, perplexity)
            lisi[:, idx] = 1 / simpson
    lisi_df = pd.DataFrame(lisi, index=meta_data.index, columns=labels_use)
    return lisi_df

if __name__ == '__main__':
    N = 100
    D = 1000
    cell_types = np.array((['typeA'] * 25) + (['typeB']*50) + (['typeC'] * 25))
    np.random.shuffle(cell_types)
    batches = np.array((['batch1'] * 40) + (['batch2']*60))
    np.random.shuffle(batches)
    print(cell_types.shape)
    print(batches.shape)
    meta_data = pd.DataFrame({'cell_type': cell_types, 'batch_ID': batches})
    X = np.random.random((N, D))
    lisi1_df = lisi(X, meta_data, labels_use=['cell_type', 'batch_ID'])
    lisi2_df = lisi2(X, meta_data, labels_use=['cell_type', 'batch_ID'])
    pass
