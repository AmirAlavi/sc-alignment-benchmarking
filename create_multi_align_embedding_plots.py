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

def load_embeddings(args):
    #results_by_task = defaultdict(list)
    embeddings = []
    for filename in glob.iglob(join(args.root_folder, f'**/plot_aligned_embedding_kwargs_*_{args.embedding}.pkl'), recursive=True):
        filename = Path(filename)
        print(filename)
        embedding_items = {}
        with open(filename, 'rb') as f:
            embedding_items['plot_kwargs'] = pickle.load(f)
        with open(filename.parent / 'results.pickle', 'rb') as f:
            results = pickle.load(f)
            embedding_items['results'] = results
            embedding_items['task'] = results['alignment_task']
        embeddings.append(embedding_items)
    return embeddings

def load_embeddings_df(args):
    task_path = []
    task_title = []
    leaveOut = []
    method = []
    dataset = []
    celltype = []
    batch = []
    x1 = []
    x2 = []
    task_subsample_idx = {}
    for filename in glob.iglob(join(args.root_folder, f'**/plot_aligned_embedding_kwargs_*{args.embedding}.pkl'), recursive=True):
        filename = Path(filename)
        print(filename)
        with open(filename, 'rb') as f:
            plot_kwargs = pickle.load(f)
            n_points = len(plot_kwargs['cell_labels'])
            with open(filename.parent / 'results.pickle', 'rb') as f:
                results = pickle.load(f)
            cur_method = results['method']
            task = results['alignment_task']
            if args.rename_method is not None:
                cur_method = rename_method(cur_method, args.rename_method)
            if args.rename_dataset is not None:
                task = rename_dataset(task, args.rename_dataset)
            task.source_batch = task.source_batch.replace('Chromium ', '')
            task.target_batch = task.target_batch.replace('Chromium ', '')
            task_key = str(task)
            if n_points > 500:
                if task_key not in task_subsample_idx:
                    idx = np.random.choice(n_points, size=500, replace=False)
                    task_subsample_idx[task_key] = idx
                else:
                    idx = task_subsample_idx[task_key]
                n_points = 500
            else:
                idx = np.arange(n_points)
            celltype.extend(plot_kwargs['cell_labels'][idx])
            batch.extend(plot_kwargs['batch_labels'][idx])
            x1.extend(plot_kwargs['embedding'][idx,0])
            x2.extend(plot_kwargs['embedding'][idx,1])
            method.extend([cur_method] * n_points)
            task_path.extend([task.as_path()] * n_points)
            task_title.extend([task.as_plot_string()] * n_points)
            leaveOut.extend([task.leave_out_ct] * n_points)
            dataset.extend([task.ds_key] * n_points)
    df = pd.DataFrame(data={'task_path': task_path, 'task': task_title, 'dataset': dataset, 'leaveOut': leaveOut, 'method': method, 'Cell type': celltype, 'Batch': batch, f'{args.embedding}1': x1, f'{args.embedding}2': x2})
    return df

def load_embeddings_by_task(args):
    # results_by_task = defaultdict(list)
    # for filename in glob.iglob(join(args.root_folder, '**/results.pickle'), recursive=True):
    #     print(filename)
    #     with open(filename, 'rb') as f:
    #         result = pickle.load(f)
    #         results_by_task[str(result['alignment_task'])].append(result)


    embeddings_by_task = defaultdict(list)
    # embeddings = []
    for filename in glob.iglob(join(args.root_folder, f'**/plot_aligned_embedding_kwargs_*_{args.embedding}.pkl'), recursive=True):
        filename = Path(filename)
        print(filename)
        embedding_items = {}
        with open(filename, 'rb') as f:
            embedding_items['plot_kwargs'] = pickle.load(f)
        with open(filename.parent / 'results.pickle', 'rb') as f:
            results = pickle.load(f)
            embedding_items['results'] = results
            # embedding_items['task'] = results['alignment_task']
            task_key = str(results['alignment_task'])
        embeddings_by_task[task_key].append(embedding_items)
    return embeddings_by_task

def plot_aligned_embedding(**kwargs):
    if 'log_dir' not in kwargs:
        raise RuntimeError('Missing kwarg log_dir')
    if 'embed_name' not in kwargs:
        raise RuntimeError('Missing kwarg embed_name')
    if 'embedding' not in kwargs:
        raise RuntimeError('Missing kwarg embedding')
    if 'cell_labels' not in kwargs:
        raise RuntimeError('Missing kwarg cell_labels')
    if 'batch_labels' not in kwargs:
        raise RuntimeError('Missing kwarg batch_labels')
    if 'task' not in kwargs:
        raise RuntimeError('Missing kwarg task')
    log_dir = kwargs['log_dir']
    task_path = kwargs['task'].as_path()
    plt.figure()
    plt.title('Color by batch', fontsize='small')
    batch_colors = ['m', 'c']
    for batch, color in zip(np.unique(kwargs['batch_labels']), batch_colors):
        idx = np.where(kwargs['batch_labels'] == batch)[0]
        plt.scatter(kwargs['embedding'][idx, 0], kwargs['embedding'][idx, 1], c=color, label=batch, alpha=0.3)
    plt.savefig(join(log_dir, f'{task_path}_{kwargs["embed_name"]}_by_batch.png'))
    plt.savefig(join(log_dir, f'{task_path}_{kwargs["embed_name"]}_by_batch.pdf'))
    plt.savefig(join(log_dir, f'{task_path}_{kwargs["embed_name"]}_by_batch.svg'))
    plt.close()
    # Plot, coloring by the cell type
    plt.figure()
    plt.title('Color by cell type', fontsize='small')
    for ct in np.unique(kwargs['cell_labels']):
        idx = np.where(kwargs['cell_labels'] == ct)[0]
        plt.scatter(kwargs['embedding'][idx, 0], kwargs['embedding'][idx, 1], label=ct, alpha=0.3)
    plt.savefig(join(log_dir, f'{task_path}_{kwargs["embed_name"]}_by_celltype.png'))
    plt.savefig(join(log_dir, f'{task_path}_{kwargs["embed_name"]}_by_celltype.pdf'))
    plt.savefig(join(log_dir, f'{task_path}_{kwargs["embed_name"]}_by_celltype.svg'))
    plt.close()

row_order = ['SCIPR-gdy', 'SCIPR-mnn', 'ScAlign', 'SeuratV3', 'MNN', 'None']

if __name__ == '__main__':
    parser = argparse.ArgumentParser('multi-alignment embeddings', description='Create publication embedding plots')
    parser.add_argument('root_folder', help='Root folder to search for result files.')
    parser.add_argument('output_folder', help='Path of output folder (created if not exists) to store plots in.')
    parser.add_argument('--filter_method', help='Filter to only look at this method data')
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
    df = pd.DataFrame({'filename':[], 'dataset':[], 'method':[], 'source_batch':[], 'target_batch':[], 'target_leave_out':[]})
    for filename in glob.iglob(join(args.root_folder, f'**/results.pickle'), recursive=True):
        filename = Path(filename)
        with open(filename, 'rb') as f:
            results = pickle.load(f)
            print(filename)
            df = df.append({
                'filename': filename,
                'dataset': results['alignment_task'].ds_key,
                'method': results['method'],
                'source_batch': results['alignment_task'].source_batch,
                'target_batch': results['alignment_task'].target_batch,
                'target_leave_out': results['alignment_task'].leave_out_ct
                }, ignore_index=True)
    print(f"found {df.shape[0]} files")
    print("filtering out the 'leave-out' experiments...")
    df = df[pd.isnull(df.target_leave_out)]
    dups = df.duplicated(subset=['dataset', 'method', 'source_batch', 'target_batch', 'target_leave_out']).sum()
    print(f"{df.shape[0]} files remain, {dups} duplicates")
    
    
    # ---
    def check_source_batch_same(cache, batch_key, to_check):
        if batch_key in cache:
            return np.allclose(cache[batch_key], to_check)
        else:
            return True
    
    obs = pd.DataFrame({'filename':[], 'method':[], 'batch':[], 'celltype':[]})
    adata = None
    #for filename in glob.iglob(join(args.root_folder, f'**/results.pickle'), recursive=True):
    for filename in df.filename:
        filename = Path(filename)
        with open(filename, 'rb') as f:
            results = pickle.load(f)
            print(filename)
        if args.filter_method is not None and results['method'] != args.filter_method:
            continue
        try:
            with open(filename.parent / 'source_aligned_x.pkl', 'rb') as f:
                source_aligned_x = pickle.load(f)

            with open(filename.parent / 'source_y.pkl', 'rb') as f:
                source_y = pickle.load(f)
                # print(source_y)

            with open(filename.parent / 'target_x.pkl', 'rb') as f:
                target_x = pickle.load(f)

            with open(filename.parent / 'target_y.pkl', 'rb') as f:
                target_y = pickle.load(f)
                # print(target_y)
            
            target_batch = results['alignment_task'].target_batch
            print(f"{results['method']} {results['alignment_task'].source_batch}->{target_batch}")
            if adata is not None:
                concat = [adata]
            else:
                concat = []
            adata_source = anndata.AnnData(X=source_aligned_x)
            adata_source.obs['filename'] = filename
            adata_source.obs['method'] = results['method']
            adata_source.obs['celltype'] = source_y.values
            adata_source.obs['batch'] = results['alignment_task'].source_batch
            adata_source.obs['batch_type'] = 'source'
            concat.append(adata_source)
            print(adata_source.obs)
            
            if adata is not None:
                same_target = adata[(adata.obs['method'] == results['method']) & 
                                    (adata.obs['batch'] == target_batch) & 
                                    (adata.obs['batch_type'] == 'target')]
                if same_target.shape[0] > 0:
                    print("I've seen this target batch before")
                    print(same_target.shape)
                    print(target_x.shape)
                    assert(same_target.shape == target_x.shape)
                    assert(np.allclose(same_target.X, target_x))
                    print("Not adding it again")
                else:
                    adata_target = anndata.AnnData(X=target_x)
                    adata_target.obs['filename'] = filename
                    adata_target.obs['method'] = results['method']
                    adata_target.obs['celltype'] = target_y.values
                    adata_target.obs['batch'] = target_batch
                    adata_target.obs['batch_type'] = 'target'
                    concat.append(adata_target)
                    print(adata_target.obs)
                    print("Adding target batch")
            else:
                # assert(check_source_batch_same(source_data_cache, source_batch, source_aligned_x))
                adata_target = anndata.AnnData(X=target_x)
                adata_target.obs['filename'] = filename
                adata_target.obs['method'] = results['method']
                adata_target.obs['celltype'] = target_y.values
                adata_target.obs['batch'] = target_batch
                adata_target.obs['batch_type'] = 'target'
                concat.append(adata_target)
                print(adata_target.obs)
                print("Adding target batch")
            
            
            adata = anndata.AnnData.concatenate(*concat, batch_key='concat_batch')
            print(adata.shape)
            

        except FileNotFoundError as e:
            print(e)
            continue
        
    print(adata.obs)
    fig, axs = plt.subplots(3, 2, figsize=(10, 21))
    for m in adata.obs['method'].unique():
        print(m)
        subset = adata[adata.obs['method'] == m]
        print('embedding...')
        pca = PCA(n_components=50).fit_transform(subset.X)
        subset.obsm['X_pca'] = pca
        print('pca done.')
        subset.obsm['X_tsne'] = TSNE().fit_transform(pca)
        print('tsne done.')
        umap_ = umap.UMAP().fit_transform(pca)
        subset.obsm['X_umap'] = umap_
        print('umap done.')
        if subset.shape[0] > 1000:
            print('data large, subsampling')
            sc.pp.subsample(subset, n_obs=1000)
        for i, b in enumerate(subset.obs['batch'].unique()):
            batch = subset[subset.obs['batch'] == b]
            axs[0, 0].scatter(batch.obsm['X_pca'][:, 0], batch.obsm['X_pca'][:, 1], label=b, alpha=0.3, facecolors='none', edgecolors=f'C{i}')
            axs[1, 0].scatter(batch.obsm['X_tsne'][:, 0], batch.obsm['X_tsne'][:, 1], label=b, alpha=0.3, facecolors='none', edgecolors=f'C{i}')
            axs[2, 0].scatter(batch.obsm['X_umap'][:, 0], batch.obsm['X_umap'][:, 1], label=b, alpha=0.3, facecolors='none', edgecolors=f'C{i}')
        axs[0, 0].legend()
        for i, ct in enumerate(subset.obs['celltype'].unique()):
            cells = subset[subset.obs['celltype'] == ct]
            print(cells.shape)
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
        plt.suptitle(m)
        plt.savefig(embeddings_folder / f'multi-align_{m}.png')
        plt.savefig(embeddings_folder / f'multi-align_{m}.svg')
            
    
    
    
    
    sys.exit()
    # embeddings = load_embeddings(args)

    # for embedding_items in embeddings:
    #     plot_kwargs = embedding_items['plot_kwargs']
    #     results = embedding_items['results']
    #     task = embedding_items['task']
    #     plot_kwargs['task'] = task
    #     plot_kwargs['log_dir'] = embeddings_folder
    #     plot_aligned_embedding(**plot_kwargs)

    embeddings = load_embeddings_df(args)
    print(embeddings.shape)
    embeddings = embeddings[pd.isnull(embeddings.leaveOut)]
    print(embeddings.shape)
    def change_facet_titles(g):
        # Required workaround for set_titles, see https://github.com/mwaskom/seaborn/issues/509#issuecomment-316132303
        for row in g.axes:
            row[-1].texts = []
        return g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
    for dataset in np.unique(embeddings.dataset):
        print(dataset)
        embeddings_subset = embeddings[embeddings.dataset == dataset]
        print(embeddings_subset.shape)
        g = sns.relplot(x=f'{args.embedding}1', y=f'{args.embedding}2', col='task', row='method', row_order=row_order, hue='Batch', kind='scatter', data=embeddings_subset, alpha=0.5, facet_kws={'sharex': False, 'sharey': False, 'margin_titles': True}, palette='husl')
        g = change_facet_titles(g)
        plt.savefig(embeddings_folder / f'{dataset}_{args.embedding}_facetgrid_batch.png')
        plt.savefig(embeddings_folder / f'{dataset}_{args.embedding}_facetgrid_batch.svg')
        g = sns.relplot(x=f'{args.embedding}1', y=f'{args.embedding}2', col='task', row='method', row_order=row_order, hue='Cell type', kind='scatter', data=embeddings_subset, alpha=0.5, facet_kws={'sharex': False, 'sharey': False, 'margin_titles': True}, palette='Dark2')
        g = change_facet_titles(g)
        plt.savefig(embeddings_folder / f'{dataset}_{args.embedding}_facetgrid_celltype.png')
        plt.savefig(embeddings_folder / f'{dataset}_{args.embedding}_facetgrid_celltype.svg')



    # results_by_task = defaultdict(list)
    # for filename in glob.iglob(join(args.root_folder, '**/results.pickle'), recursive=True):
    #     print(filename)
    #     with open(filename, 'rb') as f:
    #         result = pickle.load(f)
    #         results_by_task[str(result['alignment_task'])].append(result)
            
    # for task, results in results_by_task.items():
    #     print(task)
    #     method = []
    #     metric = []
    #     score = []
    #     for r in results:
    #         method.extend([r['method']]*(2*r['lisi'].shape[0]))
    #         for col in r['lisi'].columns:
    #             metric.extend([col]*r['lisi'].shape[0])
    #             score.extend(r['lisi'][col])
    #     if args.rename_method is not None:
    #         method = rename_methods(method, args.rename_method)
    #     alignment_task = results[0]['alignment_task']
    #     if args.rename_dataset is not None:
    #         alignment_task = rename_dataset(alignment_task, args.rename_dataset)
    #     df = pd.DataFrame(data={'method': method, 'metric': metric, 'score': score})
    #     plot_lisi(df, alignment_task, lisi_folder)

