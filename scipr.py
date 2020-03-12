# iterative point set registration methods for scRNA-seq alignment

from sklearn.preprocessing import StandardScaler, normalize
import numpy as np
from scipy import spatial
from functools import partial

import matching
import transform

class AffineSCIPR(object):
    def __init__(self, n_iter=20, n_epochs_per_iter=1000, matching_algo='mnn',
                 opt='adam', lr=1e-3, input_normalization='l2',
                 frac_matches_to_keep=1.0, source_match_thresh=0.5,
                 target_match_limit=2):
        self.n_iter = n_iter
        self.n_epochs_per_iter = n_epochs_per_iter
        self.matching_algo = matching_algo
        self.opt = opt
        self.lr = lr
        self.input_normalization = input_normalization
        self.frac_matches_to_keep = frac_matches_to_keep
        self.source_match_thresh = source_match_thresh
        self.target_match_limit = target_match_limit
    
    def fit(self, A, B):
        d = A.shape[1]
        A = self.apply_input_normalization(A)
        B = self.apply_input_normalization(B)
        matching_fcn = self.get_matching_fcn()
        if self.matching_algo in ['closest', 'mnn']:
            kd_B = spatial.cKDTree(B)
        theta = None
        for i in range(self.n_iter):
            if self.matching_algo in ['closest', 'mnn']:
                a_idx, b_idx, distances = matching_fcn(A, B, kd_B)
            else:
                a_idx, b_idx, distances = matching_fcn(A, B)
            print(f'Step: {i+1}/{self.n_iter}, pairs: {len(a_idx)}, mean_dist: {np.mean(distances)}')
            theta_new, W, bias = transform.fit_transform_affine(A[a_idx], B[b_idx], optim=self.opt, lr=self.lr, epochs=self.n_epochs_per_iter)
            A = np.dot(W, A.T).T + bias
            if theta is None:
                theta = theta_new
            else:
                theta = np.dot(theta_new, theta)
        W = theta[:d, :d]
        bias = theta[:d, -1]

        self.W_ = W
        self.bias_ = bias

    def transform(self, X):
        X = self.apply_input_normalization(X)
        return np.dot(self.W_, X.T).T + self.bias_

    def apply_input_normalization(self, X):
        if self.input_normalization == 'std':
            print('Applying Standard Scaling')
            scaler = StandardScaler().fit(X)
            return scaler.transform(X)
        elif self.input_normalization == 'l2':
            print('Applying L2 Normalization')
            return normalize(X)
        elif self.input_normalization == 'log':
            print('Applying log normalization')
            return np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)
    
    def get_matching_fcn(self):
        if self.matching_algo == 'closest':
            print('Using CLOSEST matching')
            return matching.get_closest_matches
        elif self.matching_algo == 'hungarian':
            print('Using HUNGARIAN matching')
            return partial(matching.get_hungarian_matches,
                           frac_to_match=self.frac_matches_to_keep)
        elif self.matching_algo == 'greedy':
            print('Using GREEDY matching')
            return partial(matching.get_greedy_matches,
                           source_match_threshold=self.source_match_thresh,
                           target_match_limit=self.target_match_limit)
        elif self.matching_algo == 'mnn':
            print('Using MNN matching')
            return partial(matching.get_mnn_matches)
