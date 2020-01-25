from scipy import spatial
import numpy as np

def get_closest_matches(A, B, prune_matches=False):
    # if prune_matches and D_max is None:
    #     raise ValueError('D_max cannot be None if prune_matches is True')

    kd = spatial.KDTree(B)
    distances, B_indices = kd.query(A)
    A_indices = np.arange(A.shape[0])

    if prune_matches:
        # mean = np.mean(distances)
        # std = np.std(distances)
        # z_score = (distances - mean) / std
        q25, q50, q75 = np.percentile(distances, [25, 50, 75])
        iqr = q75 - q25
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        distances_filtered = []
        B_indices_filtered = []
        A_indices_filtered = []
        for i, dist in enumerate(distances):
            if dist > lower and dist < upper:
                distances_filtered.append(dist)
                B_indices_filtered.append(B_indices[i])
                A_indices_filtered.append(i)
        distances = np.array(distances_filtered)
        B_indices = np.array(B_indices_filtered)
        A_indices = np.array(A_indices_filtered)
    return A_indices, B_indices, distances
