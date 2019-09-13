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
    batch_ax.set_title('Color by batch', fontsize='small')
    batch_ax.set_xticks([])
    batch_ax.set_yticks([])
    batch_colors = ['m', 'c']
    for batch, color in zip(np.unique(adata.obs[alignment_task.batch_key]), batch_colors):
        idx = np.where(adata.obs[alignment_task.batch_key] == batch)[0]
        batch_ax.scatter(adata.obsm[embed_key][idx, 0], adata.obsm[embed_key][idx, 1], c=color, label=batch, alpha=0.15)
    fig.add_subplot(batch_ax)
    # Plot, coloring by the cell type
    ct_ax = plt.Subplot(fig, sub_grid[1])
    ct_ax.set_title('Color by cell type', fontsize='small')
    ct_ax.set_xticks([])
    ct_ax.set_yticks([])
    for ct in np.unique(adata.obs[alignment_task.ct_key]):
        idx = np.where(adata.obs[alignment_task.ct_key] == ct)[0]
        ct_ax.scatter(adata.obsm[embed_key][idx, 0], adata.obsm[embed_key][idx, 1], label=ct, alpha=0.15)
    fig.add_subplot(ct_ax)

def setup_comparison_grid_plot(alignment_task_list, method_names):
    plot_scaler = 5
    tsne_fig = plt.figure(figsize=(plot_scaler*len(alignment_task_list), plot_scaler*len(method_names)), constrained_layout=False)
    tsne_fig.suptitle('t-SNE embeddings')
    tsne_outer_grid = tsne_fig.add_gridspec(len(method_names) + 1, len(alignment_task_list) + 1)

    pca_fig = plt.figure(figsize=(plot_scaler*len(alignment_task_list), plot_scaler*len(method_names)), constrained_layout=False)
    pca_fig.suptitle('PCA projections')
    pca_outer_grid = pca_fig.add_gridspec(len(method_names) + 1, len(alignment_task_list) + 1, wspace=0.25)

    umap_fig = plt.figure(figsize=(plot_scaler*len(alignment_task_list), plot_scaler*len(method_names)), constrained_layout=False)
    umap_fig.suptitle('UMAP projections')
    umap_outer_grid = umap_fig.add_gridspec(len(method_names) + 1, len(alignment_task_list) + 1, wspace=0.25)

    lisi_fig = plt.figure(figsize=(plot_scaler*len(alignment_task_list), plot_scaler), constrained_layout=False)
    lisi_fig.suptitle('LISI scores')
    lisi_outer_grid = lisi_fig.add_gridspec(2, len(alignment_task_list))

    for i, task in enumerate(alignment_task_list):
        ax = tsne_fig.add_subplot(tsne_outer_grid[0, i + 1])
        ax.text(0.5, 0.2, task.as_title(), va="top", ha="center")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = pca_fig.add_subplot(pca_outer_grid[0, i + 1])
        ax.text(0.5, 0.2, task.as_title(), va="top", ha="center")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = umap_fig.add_subplot(umap_outer_grid[0, i + 1])
        ax.text(0.5, 0.2, task.as_title(), va="top", ha="center")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = lisi_fig.add_subplot(lisi_outer_grid[0, i])
        ax.text(0.5, 0.3, task.as_title(), va="top", ha="center")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
    for i, method in enumerate(method_names):
        ax = tsne_fig.add_subplot(tsne_outer_grid[i+1, 0])
        if method is None:
            method = 'none'
        ax.text(0.7, 0.5, method, va="center", ha="left")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = pca_fig.add_subplot(pca_outer_grid[i+1, 0])
        if method is None:
            method = 'none'
        ax.text(0.7, 0.5, method, va="center", ha="left")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = umap_fig.add_subplot(umap_outer_grid[i+1, 0])
        if method is None:
            method = 'none'
        ax.text(0.7, 0.5, method, va="center", ha="left")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    
    return tsne_fig, tsne_outer_grid, pca_fig, pca_outer_grid, umap_fig, umap_outer_grid, lisi_fig, lisi_outer_grid