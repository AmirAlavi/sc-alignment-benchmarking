import networkx as nx
import scipy
import numpy as np
import time

np.random.seed(1234)

n_nodes = 50
frac = 0.5
dist_mat = np.random.rand(n_nodes, n_nodes)
n_to_match = int(frac * n_nodes)
t0= time.time()
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
print('done')
print(time.time() - t0)