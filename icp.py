from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython import display

activations = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU
}

# Code for ICP methods
def isnan(x):
    return x != x

def get_rigid_transformer(ndims, act=None, bias=False):
    model = nn.Sequential(nn.Linear(ndims, ndims, bias=bias))
    #print(model[0].weight.data.dtype)
    model[0].weight.data.copy_(torch.eye(ndims))#, dtype=torch.double))
    if act is not None:
        model.add_module('transformer_{}'.format(act), activations[act]())
    #print(model[0].weight.data.dtype)
    return model

def get_2_layer_rigid_transformer(ndims, act=None, bias=False):
    model = nn.Sequential(nn.Linear(ndims, ndims, bias=bias))
    #print(model[0].weight.data.dtype)
    model[0].weight.data.copy_(torch.eye(ndims))#, dtype=torch.double))
    if act is not None:
        model.add_module('transformer_{}'.format(act), activations[act]())
    model.add_module('transformer_layer2_lin', nn.Linear(ndims, ndims, bias=bias))
    model[-1].weight.data.copy_(torch.eye(ndims))
    #print(model[0].weight.data.dtype)
    return model

def closest_point_loss(A, B):
    loss = A.unsqueeze(1) - B.unsqueeze(0)
    loss = loss**2
    loss = loss.mean(dim=-1)
    loss, matches = loss.min(dim=1)
    unique_matches = np.unique(matches.numpy())
    loss = loss.sum()
    return loss, unique_matches

def closest_point_loss_ignore(A, B):
    # build distance matrix
    loss = A.unsqueeze(1) - B.unsqueeze(0)
    loss = loss**2
    loss = loss.mean(dim=-1)
    # Select pairs, iteratively building up a mask
    mask = np.zeros(loss.shape, dtype=np.float32)
    sorted_idx = np.stack(np.unravel_index(np.argsort(loss.detach().numpy().ravel()), loss.shape), axis=1)
    target_matched = set()
    source_matched = set()
    matched = 0
    for i in range(sorted_idx.shape[0]):
#         if matched >= 100:
#             break
        match_idx = sorted_idx[i]
        #if match_idx[0] not in source_matched and match_idx[1] not in target_matched:
        if match_idx[1] not in target_matched:
            mask[match_idx[0], match_idx[1]] = 1
            matched += 1
            target_matched.add(match_idx[1])
            source_matched.add(match_idx[0])
    mask = torch.from_numpy(mask)
    #print(mask.sum())
    loss = torch.mul(loss, mask).sum()
    loss = loss.sum()
    #print(target_matched)
    #sys.exit()
    return loss, np.array(list(target_matched))

def relaxed_match_loss(A, B, source_match_threshold=1.0):
    # build distance matrix
    loss = A.unsqueeze(1) - B.unsqueeze(0)
    loss = loss**2
    loss = loss.mean(dim=-1)
    # Select pairs, iteratively building up a mask
    mask = np.zeros(loss.shape, dtype=np.float32)
    sorted_idx = np.stack(np.unravel_index(np.argsort(loss.detach().numpy().ravel()), loss.shape), axis=1)
    target_matched_counts = defaultdict(int)
    source_matched = set()
    matched = 0
    for i in range(sorted_idx.shape[0]):
#         if matched >= 100:
#             break
        match_idx = sorted_idx[i]
        #if match_idx[0] not in source_matched and match_idx[1] not in target_matched:
        if target_matched_counts[match_idx[1]] < 2 and match_idx[0] not in source_matched:
            target_matched_counts[match_idx[1]] += 1
            mask[match_idx[0], match_idx[1]] = 1
            source_matched.add(match_idx[0])
        if len(source_matched) > source_match_threshold * A.shape[0]:
            break
    mask = torch.from_numpy(mask)
    #print(mask.sum())
    loss = torch.mul(loss, mask).sum()
    loss /= torch.sum(mask)
    #print(target_matched)
    #sys.exit()
    return loss, np.array(list(target_matched_counts.keys()))

def Hbeta(D, beta):
    P = torch.exp(-D * beta.clone())
    sumP = torch.sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    return H, P

def compute_Gaussian_kernel(X, tol=1e-5, perplexity=30):
    n, d = X.shape
    # dist = X.unsqueeze(1) - X.unsqueeze(0)    # creates an N x N * D difference matrix
    # dist = torch.norm(dist, p=2, dim=-1)     # L2 norm, N vector
    # dist = dist**2                          # squared L2 norm, N vector
    dist = torch.cdist(X, X, p=2)
    dist = dist**2

    P = torch.zeros((n, n))
    beta = torch.ones((n, 1))
    logU = torch.log(torch.tensor(perplexity, dtype=torch.float32))
    for i in range(n):
        betamin = -float('inf')
        betamax = float('inf')
        H, thisP = Hbeta(dist[i], beta[i])

        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta[i]
                if betamax == float('inf') or betamax == -float('inf'):
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i]
                if betamin == float('inf') or betamin == -float('inf'):
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2.
            H, thisP = Hbeta(dist[i], beta[i])
            Hdiff = H - logU
            tries += 1
        P[i, :] = thisP
    return P


""" Also adds loss terms to enforce that pairwise distances are maintained
"""
def relaxed_match_loss_xentropy(A, B, original_A, source_match_threshold=1.0):
    # build distance matrix
    mse_loss = A.unsqueeze(1) - B.unsqueeze(0) # N x N x D
    mse_loss = mse_loss**2
    mse_loss = mse_loss.mean(dim=-1) # N x N
    # Select pairs, iteratively building up a mask
    mask = np.zeros(mse_loss.shape, dtype=np.float32)
    sorted_idx = np.stack(np.unravel_index(np.argsort(mse_loss.detach().numpy().ravel()), mse_loss.shape), axis=1)
    target_matched_counts = defaultdict(int)
    source_matched = set()
    matched = 0
    for i in range(sorted_idx.shape[0]):
#         if matched >= 100:
#             break
        match_idx = sorted_idx[i]
        #if match_idx[0] not in source_matched and match_idx[1] not in target_matched:
        if target_matched_counts[match_idx[1]] < 2 and match_idx[0] not in source_matched:
            target_matched_counts[match_idx[1]] += 1
            mask[match_idx[0], match_idx[1]] = 1
            source_matched.add(match_idx[0])
        if len(source_matched) > source_match_threshold * A.shape[0]:
            break
    mask = torch.from_numpy(mask)
    mse_loss = torch.mul(mse_loss, mask).sum()
    mse_loss /= torch.sum(mask)
    # Compute cross-entropy loss
    kernel_mat = compute_Gaussian_kernel(A)
    kernel_mat_original = compute_Gaussian_kernel(original_A)
    xentropy_loss = torch.sum(torch.sum(-kernel_mat * torch.log(kernel_mat_original), dim=1)) / A.shape[0]
    loss = mse_loss + xentropy_loss
    return loss, np.array(list(target_matched_counts.keys())), mse_loss, xentropy_loss

def plot_step(A, B, type_index_dict, pca, step, matched_targets, closest_points, losses):
    A = pca.transform(A)
    B = pca.transform(B)
    mask = np.zeros(B.shape[0], dtype=bool)
    mask[matched_targets] = True
    B_matched = B[mask]
    B_unmatched = B[~mask]
    fig, axes = plt.subplots(2, 2, figsize=(20,20))
    # Scatter, colored by dataset, with matching
    axes[0,0].scatter(A[:,0], A[:,1], c='m', label='source', alpha=0.15)
    axes[0,0].scatter(B_unmatched[:,0], B_unmatched[:,1], c='b', label='target', alpha=0.15)
    axes[0,0].scatter(B_matched[:,0], B_matched[:,1], c='g', label='matched target', alpha=0.75)
    axes[0,0].legend()
    # Training loss & number of target matches
    axes[0,1].plot(closest_points, label='matches')
    axes[0,1].set_xlabel('iteration')
    axes[0,1].set_ylabel('matches')
    axes[0,1].set_title('num unique matches in target set')
    axes[0,1].legend(loc='upper left')
    ax_matches = axes[0,1].twinx()
    ax_matches.set_ylabel('loss')
    ax_matches.plot(losses, c='orange', label='loss')
    ax_matches.legend(loc='upper right')
    # Scatter, colored by dataset
    axes[1,0].scatter(A[:,0], A[:,1], c='m', label='source', alpha=0.15)
    axes[1,0].scatter(B[:,0], B[:,1], c='b', label='target', alpha=0.15)
    axes[1,0].legend()
    # Scatter, colored by cell types
    combined = np.concatenate((A, B))
    for cell_type, idx in type_index_dict.items():
        axes[1,1].scatter(combined[idx, 0], combined[idx, 1], label=cell_type, alpha=0.15)
    axes[1,1].legend()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.close()
    #time.sleep(0.5)

def plot_step_xentropy(A, B, type_index_dict, pca, step, matched_targets, closest_points, losses, mse_losses, xentropy_losses):
    A = pca.transform(A)
    B = pca.transform(B)
    mask = np.zeros(B.shape[0], dtype=bool)
    mask[matched_targets] = True
    B_matched = B[mask]
    B_unmatched = B[~mask]
    fig, axes = plt.subplots(3, 2, figsize=(20,30))
    # Scatter, colored by dataset, with matching
    axes[0,0].scatter(A[:,0], A[:,1], c='m', label='source', alpha=0.15)
    axes[0,0].scatter(B_unmatched[:,0], B_unmatched[:,1], c='b', label='target', alpha=0.15)
    axes[0,0].scatter(B_matched[:,0], B_matched[:,1], c='g', label='matched target', alpha=0.75)
    axes[0,0].legend()
    # Training loss & number of target matches
    axes[0,1].plot(closest_points, label='matches')
    axes[0,1].set_xlabel('iteration')
    axes[0,1].set_ylabel('matches')
    axes[0,1].set_title('total loss & num unique matches in target set')
    axes[0,1].legend(loc='upper left')
    ax_matches = axes[0,1].twinx()
    ax_matches.set_ylabel('loss')
    ax_matches.plot(losses, c='orange', label='loss')
    ax_matches.legend(loc='upper right')

    # Other sub-losses
    axes[1,0].plot(mse_losses)
    axes[1,0].set_xlabel('iteration')
    axes[1,0].set_ylabel('loss')
    axes[1,0].set_title('MSE loss')

    axes[1,1].plot(xentropy_losses)
    axes[1,1].set_xlabel('iteration')
    axes[1,1].set_ylabel('loss')
    axes[1,1].set_title('XEntropy loss')

    # Scatter, colored by dataset
    axes[2,0].scatter(A[:,0], A[:,1], c='m', label='source', alpha=0.15)
    axes[2,0].scatter(B[:,0], B[:,1], c='b', label='target', alpha=0.15)
    axes[2,0].legend()
    # Scatter, colored by cell types
    combined = np.concatenate((A, B))
    for cell_type, idx in type_index_dict.items():
        axes[2,1].scatter(combined[idx, 0], combined[idx, 1], label=cell_type, alpha=0.15)
    axes[2,1].legend()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.close()

def ICP(A, B, type_index_dict,
        loss_function,
        n_layers=1,
        act=None,
        max_iters=100,
        sgd_steps=100,
        tolerance=1e-4,
        standardize=True, verbose=True,
        use_xentropy_loss=False,
        source_match_threshold=0.5):# TODO: this is a nasty hack for now, if use_xentropy_loss true, uses the xentropy loss instead of loss_function
    #A, B = shift_CoM(A, B)
    if standardize:
        scaler = StandardScaler().fit(np.concatenate((A, B)))
        A = scaler.transform(A)
        B = scaler.transform(B)
    combined = np.concatenate((A, B))
    pca = PCA(n_components=2).fit(combined)
    A = torch.from_numpy(A).float()
    B = torch.from_numpy(B).float()
    assert(not isnan(A).any() and not isnan(B).any())
    if n_layers == 1:
        transformer = get_rigid_transformer(A.shape[1], act=act)
    elif n_layers == 2:
        transformer = get_2_layer_rigid_transformer(A.shape[1], act=act)
    optimizer = optim.SGD(transformer.parameters(), lr=1e-3)
    transformer.train()
    hit_nan = False
    num_matches = []
    losses = []
    mse_losses = []
    xentropy_losses = []
    for i in range(max_iters):
        if hit_nan:
            break
        try:
            A_transformed = transformer(A)
            if isnan(A_transformed).any():
                print('encountered NaNs')
                print(transformer[0].weight.data)
                hit_nan = True
                break
            optimizer.zero_grad()
            if use_xentropy_loss:
                loss, unique_matches, mse_loss, xentropy_loss = relaxed_match_loss_xentropy(A_transformed, B, A, source_match_threshold=0.5)
                mse_losses.append(mse_loss.item())
                xentropy_losses.append(xentropy_loss.item())
            else:
                loss, unique_matches = loss_function(A_transformed, B)
            losses.append(loss.item())
            num_matches.append(len(unique_matches))
            if verbose:
                if use_xentropy_loss:
                    plot_step_xentropy(A_transformed.detach().numpy(),
                                       B.detach().numpy(),
                                       type_index_dict,
                                       pca,
                                       i,
                                       unique_matches,
                                       num_matches,
                                       losses,
                                       mse_losses,
                                       xentropy_losses)
                else:
                    plot_step(A_transformed.detach().numpy(), B.detach().numpy(), type_index_dict, pca, i, unique_matches, num_matches, losses)
            else:
                print(i)
            loss.backward()
            optimizer.step()
        except KeyboardInterrupt:
            break
    return transformer