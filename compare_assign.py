import networkx as nx
import scipy
from scipy.optimize import linear_sum_assignment
import numpy as np

import time
import torch

dist_mat = np.random.rand(4,4)
n_to_match = 2

t0 = time.time()
row_ind, col_ind = linear_sum_assignment(dist_mat)
mask = np.zeros(dist_mat.shape)
mask[row_ind, col_ind] = 1
print(mask)
print()
to_sort = dist_mat * mask
to_sort[np.where(mask == 0)] = float('Inf')
print(dist_mat)
print()
print(to_sort)
sorted_idx = np.stack(np.unravel_index(np.argsort(to_sort.ravel()), dist_mat.shape), axis=1)
print(sorted_idx)
row_ind, col_ind = map(list, zip(*sorted_idx[:n_to_match]))
print(row_ind, col_ind)
mask = torch.zeros(dist_mat.shape)
mask[row_ind, col_ind] = 1
print(mask)
print(f'hungarian done {time.time()-t0}')



t0 = time.time()
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
print(f'min-cost flow done {time.time()-t0}')

