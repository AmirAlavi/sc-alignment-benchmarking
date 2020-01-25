import numpy as np

def fit_transform_rigid(A, B):
    # See http://nghiaho.com/?page_id=671
    # center
    A_centroid = np.mean(A, axis=0)
    B_centroid = np.mean(B, axis=0)

    H = np.dot((A - A_centroid).T, B - B_centroid)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    detR = np.linalg.det(R)
    x = np.identity(Vt.T.shape[1])
    x[x.shape[0]-1, x.shape[1]-1] = detR
    R = np.linalg.multi_dot([Vt.T, x, U.T])

    t = B_centroid.T - np.dot(R, A_centroid.T)

    return R, t
    

def fit_transform_affine(A, B, A_indices, B_indices):
    pass
