import networkx as nx
import scipy
import numpy as np
import pandas as pd
import time
import pickle

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
np.random.seed(1234)

# n_nodes = [5, 10, 20, 40, 80]
n_nodes = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 500, 750, 1000]
# n_nodes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# n_nodes = [5, 6, 7, 8, 9, 10]
# n_nodes = [40]
n_iterations = 5
n_vertex = []
computation_time = []
algorithm = []
n_replicate = []

for nodes in n_nodes:
    for iteration in range(n_iterations):
        print(nodes)
        #dist_mat = np.exp(np.random.rand(nodes, nodes) * 5)
        dist_mat = np.random.randint(50, size=(nodes, nodes))
        n_to_match = int(0.5 * nodes)
        #n_to_match = 1

        t0 = time.time()
        target_match_limit = 2
        source_match_threshold = 0.5
        mask = np.zeros(dist_mat.shape, dtype=np.float32)
        # sort the distances by smallest->largest
        sorted_idx = np.stack(np.unravel_index(np.argsort(dist_mat.ravel()), dist_mat.shape), axis=1)
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
        # A_indices, B_indices = np.where(mask == 1)
        # distances = dist_mat[A_indices, B_indices]
        # return A_indices, B_indices, distances
        took = time.time() - t0
        print(f'Greedy took: {took}')
        n_vertex.append(nodes)
        n_replicate.append(iteration)
        computation_time.append(took)
        algorithm.append('Greedy')

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
        print(nx.info(G))
        flow = nx.min_cost_flow(G)
        took = time.time() - t0
        print(f'min flow took: {took}')
        n_vertex.append(nodes)
        n_replicate.append(iteration)
        computation_time.append(took)
        algorithm.append('Min Cost Flow')
        print()

plot_data = {
    'Nodes': n_vertex,
    'Time (s)': computation_time,
    'Algorithm': algorithm,
    'Replicate': n_replicate
}




# plt.figure()
# plt.plot(n_nodes, greedy_times, label='Greedy')
# plt.plot(n_nodes, flow_times, label='Min Cost Flow')
# plt.xlabel('Nodes')
# plt.ylabel('Time (s)')
# plt.legend()
# plt.savefig('flow_test.png')
df = pd.DataFrame(data=plot_data)
with open('flow_test_plot_data.pkl', 'wb') as f:
    pickle.dump(df, f)
g = sns.relplot(x="Nodes", y="Time (s)",
                hue="Algorithm",
                kind="line", data=df)
g.savefig('flow_test.png')
g = sns.relplot(x="Nodes", y="Time (s)",
                hue="Algorithm",
                kind="line", data=df)
g.fig.get_axes()[0].set_yscale("log")
g.savefig('flow_test_log.png')

fig, axs = plt.subplots(1, 2, figsize=(10,4))
sns.lineplot(x="Nodes", y="Time (s)", hue="Algorithm", data=df, ax=axs[0])
sns.lineplot(x="Nodes", y="Time (s)", hue="Algorithm", data=df, ax=axs[1])
axs[1].set_yscale('log')
axs[1].set_ylabel('')
axs[1].set_title('Log-scale')
axs[0].get_legend().remove()
plt.tight_layout()
plt.savefig('flow_test_combined.png')
plt.savefig('flow_test_combined.svg')