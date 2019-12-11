# Code for ICP methods
import math
from collections import defaultdict
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
import scipy
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from tqdm import tnrange, trange

activations = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU
}

# ----------------------------------------------------------------------------
# --------------------------UTILITY FUNCTIONS---------------------------------
# ----------------------------------------------------------------------------

def pretty_tdelta(tdelta):
    hours, rem = divmod(tdelta.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, mins, secs)

def create_summary_writer(model, data_sample, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    try:
        writer.add_graph(model, data_sample)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer

def isnan(x):
    return x != x

def Hbeta(D, beta):
    P = torch.exp(-D * beta.clone())
    sumP = torch.sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    return H, P

def compute_Gaussian_kernel_with_precision(X, precisions, device):
    n, d = X.shape
    dist = X.unsqueeze(1) - X.unsqueeze(0)
    dist = dist**2
    dist = dist.sum(dim=-1)

    P = torch.zeros((n, n), device=device)
    for i in range(n):
        H, thisP = Hbeta(dist[i], precisions[i])
        P[i, :] = thisP
    return P

def compute_Gaussian_kernel(X, tol=1e-5, perplexity=30):
    n, d = X.shape
    dist = X.unsqueeze(1) - X.unsqueeze(0)
    dist = dist**2
    dist = dist.sum(dim=-1)
    # dist = torch.cdist(X, X, p=2)
    # dist = dist**2

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
    return P, beta

# ----------------------------------------------------------------------------
# --------------------------ARCHITECTURES-------------------------------------
# ----------------------------------------------------------------------------

def get_affine_transformer(ndims, bias=False, relu_out=False, act=None):
    model = nn.Sequential()
    model.add_module('lin', nn.Linear(ndims, ndims, bias=bias))
    # The transform is initialized to be the identity transform
    model[0].weight.data.copy_(torch.eye(ndims))
    if relu_out:
        model.add_module('relu_final', activations['relu']())
    elif act is not None:
        model.add_module(f'{act}_final', activations[act]())
    return model, [0]

def get_2_layer_affine_transformer(ndims, act=None, bias=False, relu_out=False):
    model = nn.Sequential()
    model.add_module('lin_0', nn.Linear(ndims, ndims, bias=bias))
    model[0].weight.data.copy_(torch.eye(ndims))
    if act is not None:
        model.add_module('{}_0'.format(act), activations[act]())
    model.add_module('lin_1', nn.Linear(ndims, ndims, bias=bias))
    model[-1].weight.data.copy_(torch.eye(ndims))
    last_lin_layer_idx = len(model)-1
    if relu_out:
        model.add_module('relu_final', activations['relu']())
    elif act is not None:
        model.add_module(f'{act}_final', activations[act]())
    return model, [0, last_lin_layer_idx]

# ----------------------------------------------------------------------------
# -------------------------LOSS FUNCTIONS-------------------------------------
# ----------------------------------------------------------------------------

def get_distance_matrix(A, B, do_mean=True):
    dist = torch.cdist(A, B, p=2)
    dist = dist**2
    if do_mean:
        dist /= A.shape[1]
    return dist

def assign_closest_points(dist_mat):
    _, idx = dist_mat.min(dim=1, keepdim=True)
    return torch.zeros_like(dist_mat).scatter_(1, idx, 1) # converts the indices of the closest points to one-hot binary masks

def assign_greedy(dist_mat, source_match_threshold=1.0, target_match_limit=2):
    # Select pairs that should be matched between set A and B,
    # iteratively building up a mask that selects those matches
    mask = np.zeros(dist_mat.shape, dtype=np.float32)
    # sort the distances by smallest->largest
    t0 = datetime.datetime.now()
    sorted_idx = np.stack(np.unravel_index(np.argsort(dist_mat.detach().numpy().ravel()), dist_mat.shape), axis=1)
    t1 = datetime.datetime.now()
    time_str = pretty_tdelta(t1 - t0)
    #print('sorting took ' + time_str)
    target_matched_counts = defaultdict(int)
    source_matched = set()
    matched = 0
    for i in range(sorted_idx.shape[0]):
        match_idx = sorted_idx[i] # A tuple, match_idx[0] is index of the pair in set A, match_idx[1] " " B
        if target_matched_counts[match_idx[1]] < target_match_limit and match_idx[0] not in source_matched:
            # if the target point in this pair hasn't been matched to too much, and the source point in this
            # pair has never been matched to, then select this pair
            mask[match_idx[0], match_idx[1]] = 1
            target_matched_counts[match_idx[1]] += 1
            source_matched.add(match_idx[0])
        if len(source_matched) > source_match_threshold * dist_mat.shape[0]:
            # if matched enough of the source set, then stop
            break
    return torch.from_numpy(mask)

def assign_hungarian(dist_mat, n_to_match):
    dist_mat = dist_mat.detach().numpy()
    row_ind, col_ind = linear_sum_assignment(dist_mat)
    mask = np.zeros_like(dist_mat)
    mask[row_ind, col_ind] = 1
    if n_to_match < dist_mat.shape[0]:
        to_sort = dist_mat * mask
        to_sort[np.where(mask == 0)] = float('Inf')
        sorted_idx = np.stack(np.unravel_index(np.argsort(to_sort.ravel()), dist_mat.shape), axis=1)
        row_ind, col_ind = map(list, zip(*sorted_idx[:n_to_match]))
        mask = np.zeros_like(dist_mat)
        mask[row_ind, col_ind] = 1
    return torch.from_numpy(mask)

def assign_bipartite_flow(dist_mat, n_to_match):
    import networkx as nx
    import scipy
    G = nx.bipartite.from_biadjacency_matrix(scipy.sparse.csr_matrix(dist_mat), nx.DiGraph)
    source_nodes = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
    target_nodes = set(G) - source_nodes
    G.add_node('s', demand = -1*n_to_match)
    G.add_node('t', demand = n_to_match)
    for n in source_nodes:
        G.add_edge('s', n, weight=0)
    for n in target_nodes:
        G.add_edge(n, 't', weight=0)
    nx.set_edge_attributes(G, 1, 'capacity')
    flow = nx.min_cost_flow(G)
    # TODO: convert flow assignment to binary masking matrix
    

def closest_point_loss(A, B):
    # loss = A.unsqueeze(1) - B.unsqueeze(0)
    # loss = loss**2
    # loss = loss.mean(dim=-1)
    loss = torch.cdist(A, B, p=2)
    loss = loss**2
    loss /= A.shape[1] # For the "mean" part of MSE
    loss, target_matches = loss.min(dim=1)
    unique_matches = np.unique(target_matches.numpy())
    loss = loss.sum()
    return loss, unique_matches

def relaxed_match_loss(A, B, source_match_threshold=1.0, target_match_limit=2, do_mean=True):
    # build distance matrix
    t0 = datetime.datetime.now()
    loss = torch.cdist(A, B, p=2)
    t1 = datetime.datetime.now()
    time_str = pretty_tdelta(t1 - t0)
    #print('cdist took ' + time_str)
    loss = loss**2
    if do_mean:
        loss /= A.shape[1]
    # Select pairs that should be matched between set A and B,
    # iteratively building up a mask that selects those matches
    mask = np.zeros(loss.shape, dtype=np.float32)
    # sort the distances by smallest->largest
    t0 = datetime.datetime.now()
    sorted_idx = np.stack(np.unravel_index(np.argsort(loss.detach().numpy().ravel()), loss.shape), axis=1)
    t1 = datetime.datetime.now()
    time_str = pretty_tdelta(t1 - t0)
    #print('sorting took ' + time_str)
    target_matched_counts = defaultdict(int)
    source_matched = set()
    matched = 0
    for i in range(sorted_idx.shape[0]):
        match_idx = sorted_idx[i] # A tuple, match_idx[0] is index of the pair in set A, match_idx[1] " " B
        if target_matched_counts[match_idx[1]] < target_match_limit and match_idx[0] not in source_matched:
            # if the target point in this pair hasn't been matched to too much, and the source point in this
            # pair has never been matched to, then select this pair
            mask[match_idx[0], match_idx[1]] = 1
            target_matched_counts[match_idx[1]] += 1
            source_matched.add(match_idx[0])
        if len(source_matched) > source_match_threshold * A.shape[0]:
            # if matched enough of the source set, then stop
            break
    mask = torch.from_numpy(mask)
    loss = torch.mul(loss, mask).sum()
    loss /= torch.sum(mask)
    return loss, np.array(list(target_matched_counts.keys()))

""" Also adds loss terms to enforce that pairwise distances are maintained
"""
def xentropy_loss(A, original_A_kernel, precisions, device):
    # Compute cross-entropy loss
    #kernel_mat, _ = compute_Gaussian_kernel(A)
    kernel_mat = compute_Gaussian_kernel_with_precision(A, precisions, device)
    #kernel_mat_original = compute_Gaussian_kernel(original_A)
    safe_log = torch.log(torch.max(original_A_kernel, torch.tensor(1e-9, dtype=torch.float32, device=device)))
    xentropy_loss = torch.sum(torch.sum(-kernel_mat * safe_log, dim=1)) / A.shape[0]
    return xentropy_loss

def plot_step_tboard(tboard, A, B, type_index_dict, pca, step, matched_targets):
    #print(f'Matched targets: {matched_targets}')
    A_pca = pca.transform(A)
    B_pca = pca.transform(B)
    # Scatter, colored by dataset, with matching
    mask = np.zeros(B_pca.shape[0], dtype=bool)
    mask[matched_targets] = True
    B_matched = B_pca[mask]
    B_unmatched = B_pca[~mask]
    fig = plt.figure()
    plt.scatter(A_pca[:,0], A_pca[:,1], c='m', label='source', alpha=0.15)
    plt.scatter(B_unmatched[:,0], B_unmatched[:,1], c='b', label='target', alpha=0.15)
    plt.scatter(B_matched[:,0], B_matched[:,1], c='g', label='matched target', alpha=0.75)
    plt.legend()
    tboard.add_figure('train/pca_matches', fig, step)
    # Scatter, colored by dataset
    fig = plt.figure()
    plt.scatter(A_pca[:,0], A_pca[:,1], c='m', label='source', alpha=0.15)
    plt.scatter(B_pca[:,0], B_pca[:,1], c='b', label='target', alpha=0.15)
    plt.legend()
    tboard.add_figure('train/pca_datasets', fig, step)
    # Scatter, colored by cell types
    fig = plt.figure()
    combined = np.concatenate((A_pca, B_pca))
    for cell_type, idx in type_index_dict.items():
        plt.scatter(combined[idx, 0], combined[idx, 1], label=cell_type, alpha=0.15)
    plt.legend()
    tboard.add_figure('train/pca_labels', fig, step)
    # t-SNE plots
    fig = plt.figure()
    combined = TSNE(n_components=2).fit_transform(np.concatenate((A, B)))
    A_size = A.shape[0]
    plt.scatter(combined[:A_size,0], combined[:A_size,1], c='m', label='source', alpha=0.15)
    plt.scatter(combined[A_size:,0], combined[A_size:,1], c='b', label='target', alpha=0.15)
    plt.legend()
    tboard.add_figure('train/tsne_datasets', fig, step)
    fig = plt.figure()
    for cell_type, idx in type_index_dict.items():
        plt.scatter(combined[idx, 0], combined[idx, 1], label=cell_type, alpha=0.15)
    plt.legend()
    tboard.add_figure('train/tsne_labels', fig, step)

def plot_tsne_tboard(tboard, A, B, type_index_dict):
    fig = plt.figure()
    combined = TSNE(n_components=2).fit_transform(np.concatenate((A, B)))
    A_size = A.shape[0]
    plt.scatter(combined[:A_size,0], combined[:A_size,1], c='m', label='source', alpha=0.15)
    plt.scatter(combined[A_size:,0], combined[A_size:,1], c='b', label='target', alpha=0.15)
    plt.legend()
    tboard.add_figure('train/tsne_original_datasets', fig)
    fig = plt.figure()
    for cell_type, idx in type_index_dict.items():
        plt.scatter(combined[idx, 0], combined[idx, 1], label=cell_type, alpha=0.15)
    plt.legend()
    tboard.add_figure('train/tsne_original_labels', fig)

def log_hparams(tboard, other_args, working_dir,
        mse_loss_function,
        n_layers=1,
        bias=False,
        act=None,
        l2_reg=0.,
        epochs=100,
        lr=1e-3,
        momentum=0.9,
        standardize=True,
        xentropy_loss_weight=0.,
        plot_every_n_steps=10):
    hp = {
        'log_dir': working_dir,
        'method_name': other_args.method,
        'mse_loss_fcn': str(mse_loss_function),
        'source_match_threshold': other_args.source_match_thresh,
        'n_layers': n_layers,
        'bias': bias,
        'act': act,
        'l2_reg': l2_reg,
        'epochs': epochs,
        'lr': lr,
        'momentum': momentum,
        'standardize': standardize,
        'xentropy_loss_weight': xentropy_loss_weight,
        'plot_every_n': plot_every_n_steps
    }
    tboard.hparams(hp, {'loss': None})
    
def ICP(A, B, type_index_dict,
        working_dir,
        mse_loss_function,
        n_layers=1,
        bias=False,
        act=None,
        l2_reg=0.,
        epochs=100,
        lr=1e-3,
        momentum=0.9,
        standardize=True,
        xentropy_loss_weight=0.,
        plot_every_n_steps=10):
    print('Looking for GPU to use...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device {}'.format(device))
    if standardize:
        scaler = StandardScaler().fit(np.concatenate((A, B)))
        A = scaler.transform(A)
        B = scaler.transform(B)
    # Fit a PCA model on the original data and use this same model for all
    # PCA visualizations so that we have a constant coordinate system to track changes in
    combined = np.concatenate((A, B))
    pca = PCA(n_components=2).fit(combined)
    # Prepare for processing by pytorch
    A = torch.from_numpy(A).float()
    B = torch.from_numpy(B).float()
    assert(not isnan(A).any() and not isnan(B).any())
    # Get transformer (a neural net)
    if n_layers == 1:
        transformer, lin_layer_indices = get_affine_transformer(A.shape[1], bias=bias)
    elif n_layers == 2:
        transformer, lin_layer_indices = get_2_layer_affine_transformer(A.shape[1], act=act, bias=bias)
    print(transformer)
    tboard = create_summary_writer(transformer, A[0], working_dir)
    # when supported, call, log_hparams here
    transformer.to(device)

    # Plot the original data in tensorboard for quick visual comparison:
    plot_tsne_tboard(tboard, A.detach().numpy(), B.detach().numpy(), type_index_dict)

    optimizer = optim.SGD(transformer.parameters(), lr=lr, momentum=momentum, weight_decay=l2_reg)
    transformer.train()
    prev_transformed = A
    if xentropy_loss_weight > 0:
        # Compute the Gaussian kernel for the original data once, reuse later
        A_kernel, precisions = compute_Gaussian_kernel(A)
    t0 = datetime.datetime.now()
    for i in trange(epochs):
        try:
            for idx, lin_idx in enumerate(lin_layer_indices):
                if isnan(transformer[lin_idx].weight).any():
                    print('encountered NaNs in weights')
                    break
                tboard.add_histogram('weights/lin_{}'.format(idx), values=transformer[lin_idx].weight.flatten(), global_step=i, bins='auto')
            A_transformed = transformer(A)
            mean_shift_norm = torch.norm(A_transformed - prev_transformed, p=1, dim=1).mean()
            tboard.add_scalar('training/mean_shift_norm', mean_shift_norm, i)
            prev_transformed = A_transformed
            if isnan(A_transformed).any():
                print('encountered NaNs in data')
                print(transformer[0].weight.data)
                break
            optimizer.zero_grad()
            total_loss = torch.tensor(0.)
            mse_loss, unique_target_matches = mse_loss_function(A_transformed, B)
            tboard.add_scalar('training/mse_loss', mse_loss.item(), i)
            tboard.add_scalar('training/uniq_targets_matched', len(unique_target_matches), i)
            total_loss += mse_loss
            if xentropy_loss_weight > 0:
                source_xentropy_loss = xentropy_loss(A_transformed, A_kernel, precisions)
                tboard.add_scalar('training/xentropy_loss', source_xentropy_loss.item(), i)
                total_loss += xentropy_loss_weight * source_xentropy_loss
            tboard.add_scalar('training/total_loss', total_loss.item(), i)
            if i % plot_every_n_steps == 0:
                plot_step_tboard(tboard, A_transformed.detach().numpy(), B.detach().numpy(), type_index_dict, pca, i, unique_target_matches)
            total_loss.backward()
            optimizer.step()
        except KeyboardInterrupt:
            break
    t1 = datetime.datetime.now()
    time_str = pretty_tdelta(t1 - t0)
    print('Training took ' + time_str)
    return transformer

def train_transform(transformer, A, B, device, correspondence_mask, kernA, kernA_precisions, max_epochs, xentropy_loss_weight, lr, momentum, l2_reg, tboard, global_step=0):
    optimizer = optim.SGD(transformer.parameters(), lr=lr, momentum=momentum, weight_decay=l2_reg)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=False)
    transformer.train()
    global_step += 1
    if xentropy_loss_weight > 0:
        print('move to device')
        A, B, correspondence_mask, kernA, kernA_precisions = A.to(device), B.to(device), correspondence_mask.to(device), kernA.to(device), kernA_precisions.to(device)
    else:
        A, B, correspondence_mask, = A.to(device), B.to(device), correspondence_mask.to(device)
    for e in trange(max_epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0., device=device)
        A_transformed = transformer(A)
        # MSE Loss
        mse_mat = get_distance_matrix(A_transformed, B)
        mse_loss = torch.mul(mse_mat, correspondence_mask).sum()
        mse_loss /= torch.sum(correspondence_mask)
        total_loss += mse_loss
        tboard.add_scalar('training/mse_loss', mse_loss.item(), global_step)
        # Cross-entropy loss
        if xentropy_loss_weight > 0:
            source_xentropy_loss = xentropy_loss(A_transformed, kernA, kernA_precisions, device)
            total_loss += xentropy_loss_weight * source_xentropy_loss
            tboard.add_scalar('training/xentropy_loss', source_xentropy_loss.item(), global_step)
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
        tboard.add_scalar('training/total_loss', total_loss.item(), global_step)
        def get_cur_lr(my_optimizer):
            for param_group in my_optimizer.param_groups:
                return param_group['lr']
        # print(f'Current LR = {get_cur_lr(optimizer)}')
        # if xentropy_loss_weight > 0:
        #     print(f'\tMSE Loss = {mse_loss.item():.4} Xentropy Loss = {source_xentropy_loss.item():.4} Total Loss = {total_loss.item():.4}')
        # else:
        #     print(f'\tTotal Loss = {total_loss.item():.4}')         
        tboard.add_scalar('training/lr', get_cur_lr(optimizer), global_step)
        global_step += 1

def ICP_converge(A, B, type_index_dict,
                 working_dir,
                 assignment_fn,
                 enforce_pos,
                 n_layers=1,
                 bias=False,
                 act=None,
                 l2_reg=0.,
                 max_steps=50,
                 tolerance=1e-2,
                 max_epochs=200,
                 lr=1e-3,
                 momentum=0.9,
                 standardize=True,
                 xentropy_loss_weight=0.,
                 plot_every_n_steps=10):
    print('Looking for GPU to use...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device {}'.format(device))
    if standardize:
        scaler = StandardScaler().fit(np.concatenate((A, B)))
        A = scaler.transform(A)
        B = scaler.transform(B)
    # Fit a PCA model on the original data and use this same model for all
    # PCA visualizations so that we have a constant coordinate system to track changes in
    combined = np.concatenate((A, B))
    pca = PCA(n_components=2).fit(combined)
    # Prepare for processing by pytorch
    A = torch.from_numpy(A).float()
    B = torch.from_numpy(B).float()
    assert(not isnan(A).any() and not isnan(B).any())
    # Get transformer (a neural net)
    if n_layers == 1:
        transformer, lin_layer_indices = get_affine_transformer(A.shape[1], bias=bias, relu_out=enforce_pos, act=act)
    elif n_layers == 2:
        transformer, lin_layer_indices = get_2_layer_affine_transformer(A.shape[1], act=act, bias=bias, relu_out=enforce_pos)
    print(transformer)
    tboard = create_summary_writer(transformer, A[0], working_dir)
    # when supported, call, log_hparams here
    transformer.to(device)

    # Plot the original data in tensorboard for quick visual comparison:
    plot_tsne_tboard(tboard, A.detach().cpu().numpy(), B.detach().cpu().numpy(), type_index_dict)

    prev_transformed = A
    A_kernel = None
    precisions = None
    if xentropy_loss_weight > 0:
        # Compute the Gaussian kernel for the original data once, reuse later
        A_kernel, precisions = compute_Gaussian_kernel(A)
    t0 = datetime.datetime.now()
    for i in range(max_steps):
        try:
            print(f'Step {i}') 
            # do matching
            # for idx, lin_idx in enumerate(lin_layer_indices):
            #     if isnan(transformer[lin_idx].weight).any():
            #         print('encountered NaNs in weights')
            #         break
            #     print('adding histogram')
            #     tboard.add_histogram('weights/lin_{}'.format(idx), values=transformer[lin_idx].weight.flatten(), global_step=i, bins='auto')
            print('transforming...')
            A_transformed = transformer(A.to(device)).cpu()
            print('done.')
            print('computing mean shift norm')
            mean_shift_norm = torch.norm(A_transformed - prev_transformed, p=1, dim=1).mean()
            print(f'shift: {mean_shift_norm.item()}')
            if mean_shift_norm <= tolerance and i > 0:
                print(f'Stopping criterion satisfied, data shift norm mean = {mean_shift_norm.item()} <= {tolerance}')
                break
            print('adding scalar')
            tboard.add_scalar('training/mean_shift_norm', mean_shift_norm, i)
            prev_transformed = A_transformed
            if isnan(A_transformed).any():
                print('encountered NaNs in data')
                print(transformer[0].weight.data)
                break
            print('compute dist mat')
            dist_mat = get_distance_matrix(A_transformed, B)
            print('get assignments')
            pair_assignment_mask = assignment_fn(dist_mat)
            
            target_hits = np.unique(np.where(pair_assignment_mask.numpy() == 1)[1])
            tboard.add_scalar('training/uniq_targets_matched', len(target_hits), i)
            print('start train')
            train_transform(transformer, A, B, device, pair_assignment_mask, A_kernel, precisions, max_epochs, xentropy_loss_weight, lr, momentum, l2_reg, tboard, global_step=i*max_epochs)
            if i % plot_every_n_steps == 0:
                A_transformed = transformer(A.to(device)).cpu()
                plot_step_tboard(tboard, A_transformed.detach().numpy(), B.detach().numpy(), type_index_dict, pca, i, target_hits)
        except KeyboardInterrupt:
            break
    t1 = datetime.datetime.now()
    time_str = pretty_tdelta(t1 - t0)
    print('Training took ' + time_str)
    return transformer
