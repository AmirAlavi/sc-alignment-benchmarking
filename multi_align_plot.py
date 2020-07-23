# import pdb; pdb.set_trace()
import argparse
import os
import sys
import pickle
import glob
from os.path import join
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import scanpy as sc
sns.set_context("talk")


def rename_method(method, renames):
    for rn in renames:
        original, new = rn.split(',')
        if method == original:
            return new
    return method

def rename_methods(methods, renames):
    for i in range(len(methods)):
        methods[i] = rename_method(methods[i], renames)
    return methods

def rename_dataset(alignment_task, renames):
    for rn in renames:
        original, new = rn.split(',')
        if alignment_task.ds_key == original:
            alignment_task.ds_key = new
    return alignment_task


def plot_embedding(adata, folder_path, embedding='tsne'):
    if embedding not in ['tsne', 'pca', 'umap']:
        raise ValueError("not an available embedding")
    fig, axs = plt.subplots(1, 2, figsize=(13, 8))
    for i, b in enumerate(adata.obs['batch'].unique()):
        batch = adata[adata.obs['batch'] == b]
        axs[0].scatter(batch.obsm[f'X_{embedding}'][:, 0], batch.obsm[f'X_{embedding}'][:, 1], label=b, alpha=0.3, facecolors='none', edgecolors=f'C{i}')
    axs[0].legend()
    for i, ct in enumerate(adata.obs['celltype'].unique()):
        cells = adata[adata.obs['celltype'] == ct]
        axs[1].scatter(cells.obsm[f'X_{embedding}'][:, 0], cells.obsm[f'X_{embedding}'][:, 1], label=ct, alpha=0.3, facecolors='none', edgecolors=f'C{i}')
    axs[1].legend()
    axs[0].set_xlabel(f'{embedding.upper()} 1')
    axs[0].set_ylabel(f'{embedding.upper()} 2')
    axs[1].set_xlabel(f'{embedding.upper()} 1')
    method_name = adata.obs['method'].unique()[0]
    plt.suptitle(method_name)
    plt.savefig(folder_path / f'multi-align_{method_name}_{embedding}.png')
    plt.savefig(folder_path / f'multi-align_{method_name}_{embedding}.svg')    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('multi-alignment embeddings', description='Create publication embedding plots')
    parser.add_argument('root_folder', help='Root folder to search for result files.')
    parser.add_argument('output_folder', help='Path of output folder (created if not exists) to store plots in.')
    parser.add_argument('--embedding', help='Which type of embedding ot use', choices=['PCA', 'TSNE', 'UMAP'], default='UMAP')
    parser.add_argument('--rename_method', help='Change the text name of a particular method to appear in the plots.', action='append')
    parser.add_argument('--rename_dataset', help='Change the text name of a particular dataset to appear in the plots.', action='append')

    args = parser.parse_args()
    embeddings_folder = Path(args.output_folder) / 'multi-embeddings'
    # clf_folder = Path(args.output_folder) / 'classification'
    # kbet_folder = Path(args.output_folder) / 'kBET'
    for path in [args.output_folder, embeddings_folder]:#, clf_folder, kbet_folder]:
        if not os.path.exists(path):
            os.makedirs(path)

    for filename in glob.iglob(join(args.root_folder, f'**/aligned.h5ad'), recursive=True):
        print(filename)
        filename = Path(filename)
        adata = anndata.read_h5ad(filename)
        pca = PCA(n_components=50).fit_transform(adata.X)
        adata.obsm['X_pca'] = pca
        print('pca done.')
        adata.obsm['X_tsne'] = TSNE().fit_transform(pca)
        print('tsne done.')
        umap_ = umap.UMAP().fit_transform(pca)
        adata.obsm['X_umap'] = umap_
        print('umap done.')
        
        if adata.shape[0] > 1000:
            print('data large, subsampling')
            sc.pp.subsample(adata, n_obs=1000)

        plot_embedding(adata, embeddings_folder, 'pca')
        plot_embedding(adata, embeddings_folder, 'tsne')
        plot_embedding(adata, embeddings_folder, 'umap')
            
        fig, axs = plt.subplots(3, 2, figsize=(10, 21))
        for i, b in enumerate(adata.obs['batch'].unique()):
            batch = adata[adata.obs['batch'] == b]
            axs[0, 0].scatter(batch.obsm['X_pca'][:, 0], batch.obsm['X_pca'][:, 1], label=b, alpha=0.3, facecolors='none', edgecolors=f'C{i}')
            axs[1, 0].scatter(batch.obsm['X_tsne'][:, 0], batch.obsm['X_tsne'][:, 1], label=b, alpha=0.3, facecolors='none', edgecolors=f'C{i}')
            axs[2, 0].scatter(batch.obsm['X_umap'][:, 0], batch.obsm['X_umap'][:, 1], label=b, alpha=0.3, facecolors='none', edgecolors=f'C{i}')
        axs[0, 0].legend()
        for i, ct in enumerate(adata.obs['celltype'].unique()):
            cells = adata[adata.obs['celltype'] == ct]
            axs[0, 1].scatter(cells.obsm['X_pca'][:, 0], cells.obsm['X_pca'][:, 1], label=ct, alpha=0.3, facecolors='none', edgecolors=f'C{i}')
            axs[1, 1].scatter(cells.obsm['X_tsne'][:, 0], cells.obsm['X_tsne'][:, 1], label=ct, alpha=0.3, facecolors='none', edgecolors=f'C{i}')
            axs[2, 1].scatter(cells.obsm['X_umap'][:, 0], cells.obsm['X_umap'][:, 1], label=ct, alpha=0.3, facecolors='none', edgecolors=f'C{i}')
        axs[0, 1].legend()
        axs[0,0].set_xlabel('PCA 1')
        axs[0,0].set_ylabel('PCA 2')
        axs[0,1].set_xlabel('PCA 1')
        axs[1,0].set_xlabel('TSNE 1')
        axs[1,0].set_ylabel('TSNE 2')
        axs[1,1].set_xlabel('TSNE 1')
        axs[2,0].set_xlabel('UMAP 1')
        axs[2,0].set_ylabel('UMAP 2')
        axs[2,1].set_xlabel('UMAP 1')
        method_name = adata.obs['method'].unique()[0]
        plt.suptitle(method_name)
        plt.savefig(embeddings_folder / f'multi-align_{method_name}.png')
        plt.savefig(embeddings_folder / f'multi-align_{method_name}.svg')
