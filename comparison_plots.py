import matplotlib.pyplot as plt
import numpy as np

def plot_lisi(lisi_dfs, method_names, alignment_task, fig, figure_grid, i, j):
    sub_grid = figure_grid[i,j].subgridspec(1, 2, wspace=0.9)
    ilisi_ax = plt.Subplot(fig, sub_grid[0])
    ilisi_ax.set_title('Dataset mixing')
    lisi_data = [df[alignment_task.batch_key].values for df in lisi_dfs]
    ilisi_ax.boxplot(lisi_data, vert=False, labels=method_names, showfliers=False)
    ilisi_ax.set_xlabel('iLISI')
    fig.add_subplot(ilisi_ax)
    
    clisi_ax = plt.Subplot(fig, sub_grid[1])
    clisi_ax.set_title('Cell-type mixing')
    lisi_data = [df[alignment_task.ct_key] for df in lisi_dfs]
    clisi_ax.boxplot(lisi_data, vert=False, labels=method_names, showfliers=False)
    clisi_ax.set_xlabel('cLISI')
    fig.add_subplot(clisi_ax)

# Plotting function to visualize each alignment task result
def plot_embedding_in_grid(adata, embed_key, alignment_task, fig, figure_grid, i, j, standardize=True):
#     if standardize:
#         scaler = StandardScaler().fit(adata.obsm[embed_key])
#         new_embed_key = embed_key + '_std'
#         adata.obsm[new_embed_key] = scaler.transform(adata.obsm[embed_key])
#         embed_key = new_embed_key
    
    # Plot, coloring by the batch
    sub_grid = figure_grid[i,j].subgridspec(1, 2)
    batch_ax = plt.Subplot(fig, sub_grid[0])
    batch_ax.set_xticks([])
    batch_ax.set_yticks([])
    batch_colors = ['m', 'c']
    for batch, color in zip(np.unique(adata.obs[alignment_task.batch_key]), batch_colors):
        idx = np.where(adata.obs[alignment_task.batch_key] == batch)[0]
        batch_ax.scatter(adata.obsm[embed_key][idx, 0], adata.obsm[embed_key][idx, 1], c=color, label=batch, alpha=0.15)
    fig.add_subplot(batch_ax)
    # Plot, coloring by the cell type
    ct_ax = plt.Subplot(fig, sub_grid[1])
    ct_ax.set_xticks([])
    ct_ax.set_yticks([])
    for ct in np.unique(adata.obs[alignment_task.ct_key]):
        idx = np.where(adata.obs[alignment_task.ct_key] == ct)[0]
        ct_ax.scatter(adata.obsm[embed_key][idx, 0], adata.obsm[embed_key][idx, 1], label=ct, alpha=0.15)
    fig.add_subplot(ct_ax)

def setup_comparison_grid_plot():
    plot_scaler = 5
    fig = plt.figure(figsize=(plot_scaler*len(alignment_tasks), plot_scaler*len(methods)), constrained_layout=False)
    outer_grid = fig.add_gridspec(len(methods) + 1, len(alignment_tasks) + 1)

    pca_fig = plt.figure(figsize=(plot_scaler*len(alignment_tasks), plot_scaler*len(methods)), constrained_layout=False)
    pca_outer_grid = pca_fig.add_gridspec(len(methods) + 1, len(alignment_tasks) + 1, wspace=0.25)

    lisi_fig = plt.figure(figsize=(plot_scaler*len(alignment_tasks), plot_scaler), constrained_layout=False)
    lisi_outer_grid = lisi_fig.add_gridspec(2, len(alignment_tasks))