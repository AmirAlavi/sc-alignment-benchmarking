import pdb; pdb.set_trace()
import argparse
import os
import pickle
import glob
from os.path import join
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from submit_small_experiments import get_method_info

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
    celltype = []
    batch = []
    x1 = []
    x2 = []
    for filename in glob.iglob(join(args.root_folder, f'**/plot_aligned_embedding_kwargs_*_{args.embedding}.pkl'), recursive=True):
        filename = Path(filename)
        print(filename)
        with open(filename, 'rb') as f:
            plot_kwargs = pickle.load(f)
            n_points = len(plot_kwargs['cell_labels'])
            celltype.extend(plot_kwargs['cell_labels'])
            batch.extend(plot_kwargs['batch_labels'])
            x1.extend(plot_kwargs['embedding'][:,0])
            x2.extend(plot_kwargs['embedding'][:,1])
        with open(filename.parent / 'results.pickle', 'rb') as f:
            results = pickle.load(f)
            cur_method = results['method']
            task = results['alignment_task']
            if args.rename_method is not None:
                cur_method = rename_method(cur_method, args.rename_method)
            method.extend([cur_method] * n_points)
            if args.rename_dataset is not None:
                task = rename_dataset(task, args.rename_dataset)
            task_path.extend([task.as_path()] * n_points)
            task_title.extend([task.as_plot_string()] * n_points)
            leaveOut.extend([task.leave_out_ct] * n_points)
    df = pd.DataFrame(data={'task_path': task_path, 'task': task_title, 'leaveOut': leaveOut, 'method': method, 'celltype': celltype, 'batch': batch, f'{args.embedding}1': x1, f'{args.embedding}2': x2})
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('compile-embeddings', description='Create publication embedding plots')
    parser.add_argument('root_folder', help='Root folder to search for result files.')
    parser.add_argument('output_folder', help='Path of output folder (created if not exists) to store plots in.')
    parser.add_argument('--embedding', help='Which type of embedding ot use', choices=['PCA', 'TSNE', 'UMAP'], default='UMAP')
    parser.add_argument('--rename_method', help='Change the text name of a particular method to appear in the plots.', action='append')
    parser.add_argument('--rename_dataset', help='Change the text name of a particular dataset to appear in the plots.', action='append')

    args = parser.parse_args()
    embeddings_folder = Path(args.output_folder) / 'embeddings'
    # clf_folder = Path(args.output_folder) / 'classification'
    # kbet_folder = Path(args.output_folder) / 'kBET'
    for path in [args.output_folder, embeddings_folder]:#, clf_folder, kbet_folder]:
        if not os.path.exists(path):
            os.makedirs(path)

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
    g = sns.relplot(x=f'{args.embedding}1', y=f'{args.embedding}2', col='task', row='method', hue='batch', kind='scatter', data=embeddings, alpha=0.5, facet_kws={'margin_titles': True}, palette='husl')
    plt.savefig(embeddings_folder / 'facetgrid_batch.png')
    g = sns.relplot(x=f'{args.embedding}1', y=f'{args.embedding}2', col='task', row='method', hue='celltype', kind='scatter', data=embeddings, alpha=0.5, facet_kws={'margin_titles': True}, palette='Dark2')
    plt.savefig(embeddings_folder / 'facetgrid_celltype.png')
    sns.set_context("talk")
    g = sns.relplot(x=f'{args.embedding}1', y=f'{args.embedding}2', col='task', row='method', hue='batch', kind='scatter', data=embeddings, alpha=0.5, facet_kws={'margin_titles': True}, palette='husl')
    plt.savefig(embeddings_folder / 'facetgrid_batch_large.png')
    g = sns.relplot(x=f'{args.embedding}1', y=f'{args.embedding}2', col='task', row='method', hue='celltype', kind='scatter', data=embeddings, alpha=0.5, facet_kws={'margin_titles': True}, palette='Dark2')
    plt.savefig(embeddings_folder / 'facetgrid_celltype_large.png')



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

