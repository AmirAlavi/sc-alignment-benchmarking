# Reduce dimensions (PCA, UMAP, t-SNE) & visualize
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



def embed(datasets, key, n_pc, do_standardize):
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
    print('fitting PCA')
    pca_model = PCA(n_components=n_pc).fit(datasets[key].X)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
    ax1.bar(np.arange(n_pc) + 1, pca_model.explained_variance_ratio_)
    ax1.set_ylabel('explained variance')
    ax1.set_xlabel('PC')
    ax2.plot(np.cumsum(pca_model.explained_variance_ratio_))
    ax2.plot(np.ones_like(pca_model.explained_variance_ratio_)*0.9)
    ax2.set_xlabel('number of components')
    ax2.set_ylabel('cumulative explained variance')
    #fig.show()

    if do_standardize:
        # standardize the data for the PCA reduced dimensions (prior to fitting PCA)
        datasets[key].X = StandardScaler().fit_transform(datasets[key].X)
        print('fitting PCA (Standardized)')
        pca_model = PCA(n_components=n_pc).fit(datasets[key].X)
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
        ax1.bar(np.arange(n_pc) + 1, pca_model.explained_variance_ratio_)
        ax1.set_ylabel('explained variance')
        ax1.set_xlabel('PC')
        ax2.plot(np.cumsum(pca_model.explained_variance_ratio_))
        ax2.plot(np.ones_like(pca_model.explained_variance_ratio_)*0.9)
        ax2.set_xlabel('number of components')
        ax2.set_ylabel('cumulative explained variance')
    datasets[key].obsm['PCA'] = pca_model.transform(datasets[key].X)
    # # Data standardizer for when using PCA
    # datasets[key].uns['PCA-Standard-Scaler'] = StandardScaler.fit(datasets[key].obsm['PCA'])
    print('fitting UMAP')
    datasets[key].obsm['UMAP'] = umap.UMAP().fit_transform(datasets[key].X)
    print('fitting tSNE')
    datasets[key].obsm['TSNE'] = TSNE(n_components=2).fit_transform(datasets[key].obsm['PCA'])

def visualize(datasets, ds_key, cell_type_key='cell_type', batch_key='batch'):
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
    for cell_type, shade in zip(np.unique(datasets[ds_key].obs[cell_type_key]), ['m', 'g', 'c']):
        for batch, opacity, marker in zip(np.unique(datasets[ds_key].obs[batch_key]), opacities, markers):
            idx = np.where((datasets[ds_key].obs[cell_type_key] == cell_type) & (datasets[ds_key].obs[batch_key] == batch))[0]
            for embedding_key, ax in zip(['PCA', 'UMAP', 'TSNE'], [ax1, ax2, ax3]):
                X_subset = datasets[ds_key].obsm[embedding_key][idx, :2]
                ax.scatter(X_subset[:,0], X_subset[:,1], s=20, c=shade, edgecolors='none', marker=marker, alpha=opacity, label='{}_{}'.format(cell_type, batch))
    plt.legend(markerscale=3., loc="upper left", bbox_to_anchor=(1,1))
    plt.subplots_adjust(right=0.85)
    plt.savefig('{}_embeddings.pdf'.format(ds_key), bbox='tight')
    plt.show
