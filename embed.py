# Reduce dimensions (PCA, UMAP, t-SNE) & visualize
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def get_cmap(n, name='hsv'):
    # From https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def embed(datasets, key, n_pc, do_standardize, log_dir=None):
    """Embeds datasets via PCA, t-SNE, and UMAP

    Args:
        datasets (dict): dict of <string, AnnData>
        key (str): the key into the datasets dict of the particular
            dataset to be embedded
        n_pc (int): number of principle components for PCA projection.
        do_standardize (bool): whether to normalize to zero-mean,
            unit variance before embedding

    Side-effect:
        Adds each embedding projection of the data as an 'observation' field (obsm)
            in the datasets object.
    """
    if do_standardize:
        print('fitting PCA (Standardized)')
        X = StandardScaler().fit_transform(datasets[key].X)
        pca_model = PCA(n_components=n_pc, random_state=1373).fit(X)
        datasets[key].obsm['PCA'] = pca_model.transform(X)
    else:
        print('fitting PCA')
        pca_model = PCA(n_components=n_pc, random_state=1373).fit(datasets[key].X)
        datasets[key].obsm['PCA'] = pca_model.transform(datasets[key].X)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
    ax1.bar(np.arange(n_pc) + 1, pca_model.explained_variance_ratio_)
    ax1.set_ylabel('explained variance')
    ax1.set_xlabel('PC')
    ax2.plot(np.cumsum(pca_model.explained_variance_ratio_))
    ax2.plot(np.ones_like(pca_model.explained_variance_ratio_)*0.9)
    ax2.set_xlabel('number of components')
    ax2.set_ylabel('cumulative explained variance')
    if log_dir is not None:
        plt.savefig(log_dir / 'explained_variance_pca.png')

    print('fitting UMAP')
    datasets[key].obsm['UMAP'] = umap.UMAP().fit_transform(datasets[key].obsm['PCA'])
    print('fitting tSNE')
    datasets[key].obsm['TSNE'] = TSNE(n_components=2).fit_transform(datasets[key].obsm['PCA'])

def visualize(datasets, ds_key, cell_type_key='cell_type', batch_key='batch', log_dir=None):
    """Visualize embeddings, colored by cell type, and opacity by batch.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))
    fig.suptitle('Embeddings of Original {} Data, color = cell type'.format(ds_key))
    ax1.set_title('PCA')
    ax2.set_title('t-SNE')
    ax3.set_title('UMAP')
    num_batches = len(np.unique(datasets[ds_key].obs[batch_key]))
    opacities = [0.6, 0.2, 0.2][:num_batches]
    markers = ['o', 'o', 'P'][-num_batches:]
    cmap = get_cmap(len(np.unique(datasets[ds_key].obs[cell_type_key])), 'jet')
    for color_idx, cell_type in enumerate(np.unique(datasets[ds_key].obs[cell_type_key])):
        for batch, opacity, marker in zip(np.unique(datasets[ds_key].obs[batch_key]), opacities, markers):
            idx = np.where((datasets[ds_key].obs[cell_type_key] == cell_type) & (datasets[ds_key].obs[batch_key] == batch))[0]
            for embedding_key, ax in zip(['PCA', 'UMAP', 'TSNE'], [ax1, ax2, ax3]):
                X_subset = datasets[ds_key].obsm[embedding_key][idx, :2]
                ax.scatter(X_subset[:,0], X_subset[:,1], s=20, c=[cmap(color_idx)], edgecolors='none', marker=marker, alpha=opacity, label='{}_{}'.format(cell_type, batch))
    plt.legend(markerscale=3., loc="upper left", bbox_to_anchor=(1,1))
    plt.subplots_adjust(right=0.85)
    if log_dir is not None:
        plt.savefig(log_dir / '{}_embeddings.pdf'.format(ds_key), bbox='tight')
    # plt.savefig('{}_embeddings.pdf'.format(ds_key), bbox='tight')
    plt.show

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
    # fig.suptitle('Embeddings of Original {} Data, color = cell type'.format(ds_key))
    # ax1.set_title('PCA')
    # ax2.set_title('UMAP')
    # num_batches = len(np.unique(datasets[ds_key].obs[batch_key]))
    # opacities = [0.6, 0.2, 0.2][:num_batches]
    # markers = ['o', 'o', 'P'][-num_batches:]
    # cmap = get_cmap(len(np.unique(datasets[ds_key].obs[cell_type_key])), 'jet')
    # for color_idx, cell_type in enumerate(np.unique(datasets[ds_key].obs[cell_type_key])):
    #     for batch, opacity, marker in zip(np.unique(datasets[ds_key].obs[batch_key]), opacities, markers):
    #         idx = np.where((datasets[ds_key].obs[cell_type_key] == cell_type) & (datasets[ds_key].obs[batch_key] == batch))[0]
    #         for embedding_key, ax in zip(['PCA', 'UMAP'], [ax1, ax2]):
    #             X_subset = datasets[ds_key].obsm[embedding_key][idx, :2]
    #             ax.scatter(X_subset[:,0], X_subset[:,1], s=20, c=[cmap(color_idx)], edgecolors='none', marker=marker, alpha=opacity, label='{}_{}'.format(cell_type, batch))
    # plt.legend(markerscale=3., loc="upper left", bbox_to_anchor=(1,1))
    # plt.subplots_adjust(right=0.85)
    # if log_dir is not None:
    #     plt.savefig(log_dir / '{}_embeddings.pdf'.format(ds_key), bbox='tight')
    # # plt.savefig('{}_embeddings.pdf'.format(ds_key), bbox='tight')
    # plt.show
