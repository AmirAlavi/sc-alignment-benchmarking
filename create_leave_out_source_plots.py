from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib as mpl
import pandas as pd
import pickle
import glob
from os.path import join
from pathlib import Path
import argparse
import os
plt.rcParams['svg.fonttype'] = 'none'

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


def create_figure(df, embedding, path):
    print(df.columns)
    print(df.shape)
    source_cell_types = np.unique(df.sourceLeaveOut)
    ct_colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e'][:len(np.unique(df['Cell type']))]
    ct_color_map = {ct: color for ct, color in zip(np.unique(df['Cell type']), ct_colors)}
    batch_colors = ['#1A85FF', '#D41159', '#e6ab02'][:len(np.unique(df['Batch']))]
    batch_color_map = {batch: color for batch, color in zip(np.unique(df['Batch']), batch_colors)}
    print(ct_color_map)
    print(batch_color_map)
    def_figsize = mpl.rcParams['figure.figsize']
    figsize = [s*1.5 for s in def_figsize]
    figsize = [8.33, 10]
    #width = max(def_figsize)
    #figsize = [width*1.5, width*1.5]
    #fig, axs = plt.subplots(nrows=len(source_cell_types), ncols=2, gridspec_kw={'hspace': 0.3, 'wspace': 0.3}, figsize=figsize, subplot_kw=dict(box_aspect=1))
    fig, axs = plt.subplots(nrows=len(source_cell_types), ncols=2, subplot_kw=dict(box_aspect=1), constrained_layout=True, figsize=figsize)
    marker_size = 6.0
    alpha = 0.25
    axs[0, 0].set_title('Aligned  (missing)\ncolor=Batch')
    axs[0, 1].set_title('Aligned (all)\ncolor=CellType')
    for i, ct in enumerate(source_cell_types):
        # subset = df[(df['sourceLeaveOut'] == ct) & (df['Cell type'] != ct)]
        # b_colors = [batch_color_map[b] for b in subset['Batch']]
        # axs[i, 0].scatter(subset[f'orig_{embedding}1'], subset[f'orig_{embedding}2'], c=b_colors, alpha=alpha, s=marker_size)
        # axs[i, 0].set_ylabel(ct)
        # c_colors = [ct_color_map[ct_] for ct_ in subset['Cell type']]
        # axs[i, 1].scatter(subset[f'orig_{embedding}1'], subset[f'orig_{embedding}2'], c=c_colors, alpha=alpha, s=marker_size)
        axs[i, 0].set_ylabel(ct)
        subset = df[(df['sourceLeaveOut'] == ct) & (df['Cell type'] == ct)]
        b_colors = [batch_color_map[b] for b in subset['Batch']]
        #axs[i, 0].set(aspect='equal')
        axs[i, 0].scatter(subset[f'{embedding}1'], subset[f'{embedding}2'], c=b_colors, alpha=alpha, s=marker_size)
        subset = df[df['sourceLeaveOut'] == ct]
        c_colors = [ct_color_map[ct_] for ct_ in subset['Cell type']]
        # axs[i, 1].set(aspect='equal')
        axs[i, 1].scatter(subset[f'{embedding}1'], subset[f'{embedding}2'], c=c_colors, alpha=alpha, s=marker_size)
    legend_elements = []
    # empty_patch = mpl.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False)
    legend_elements.append(mpl.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False, label='Batch'))
    for b, c in batch_color_map.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=b, markerfacecolor=c, markersize=marker_size))
    legend_elements.append(mpl.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False, label='Cell Type'))
    for ct, c in ct_color_map.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=ct, markerfacecolor=c, markersize=marker_size))
    # fig.legend(handles=legend_elements, loc="center right")
    # plt.savefig(path.with_suffix('.png'))
    #for ax in axs.flatten():
    #    ax.set_aspect(aspect='equal')
    #plt.subplots_adjust(wspace=0., hspace=0.1)
    fig.legend(handles=legend_elements, bbox_to_anchor=(1.0, 0.45),loc = 'center left')
    plt.savefig(path.with_suffix('.png'), bbox_inches='tight')
    plt.savefig(path.with_suffix('.svg'), bbox_inches='tight')

def load_embeddings_df(args):
    ct_key = []
    batch_key = []
    task_path = []
    task_title = []
    sourceLeaveOut = []
    leaveOut = []
    method = []
    dataset = []
    source = []
    target = []
    celltype = []
    batch = []
    orig_x1 = []
    orig_x2 = []
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
            orig_x1.extend(plot_kwargs['original_embedding'][idx,0])
            orig_x2.extend(plot_kwargs['original_embedding'][idx,1])
            x1.extend(plot_kwargs['embedding'][idx,0])
            x2.extend(plot_kwargs['embedding'][idx,1])
            method.extend([cur_method] * n_points)
            task_path.extend([task.as_path()] * n_points)
            task_title.extend([task.as_plot_string()] * n_points)
            leaveOut.extend([task.leave_out_ct] * n_points)
            sourceLeaveOut.extend([task.leave_out_source_ct] * n_points)
            dataset.extend([task.ds_key] * n_points)
            source.extend([task.source_batch] * n_points)
            target.extend([task.target_batch] * n_points)
            ct_key.extend([task.ct_key] * n_points)
            batch_key.extend([task.batch_key] * n_points)
    df = pd.DataFrame(data={
        'task_path': task_path,
        'task': task_title,
        'dataset': dataset,
        'ct_key': ct_key,
        'batch_key': batch_key,
        'source': source,
        'target': target,
        'leaveOut': leaveOut,
        'sourceLeaveOut': sourceLeaveOut,
        'method': method,
        'Cell type': celltype,
        'Batch': batch,
        f'orig_{args.embedding}1': orig_x1,
        f'orig_{args.embedding}2': orig_x2,
        f'{args.embedding}1': x1,
        f'{args.embedding}2': x2})
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser('compile-embeddings', description='Create publication embedding plots')
    parser.add_argument('root_folder', help='Root folder to search for result files.')
    parser.add_argument('output_folder', help='Path of output folder (created if not exists) to store plots in.')
    parser.add_argument('--embedding', help='Which type of embedding ot use', choices=['PCA', 'TSNE', 'UMAP'], default='UMAP')
    parser.add_argument('--rename_method', help='Change the text name of a particular method to appear in the plots.', action='append')
    parser.add_argument('--rename_dataset', help='Change the text name of a particular dataset to appear in the plots.', action='append')

    args = parser.parse_args()
    embeddings_folder = Path(args.output_folder) / 'sourceLeaveOutembeddings'
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

    for ds in np.unique(embeddings.dataset):
        ds_subset = embeddings[embeddings.dataset == ds]
        for m in np.unique(ds_subset.method):
            m_subset = ds_subset[ds_subset.method == m]
            for b in np.unique(m_subset.source):
                b_subset = m_subset[m_subset.source == b]
                print(f'{ds} {m} {b}')
                create_figure(b_subset, args.embedding, embeddings_folder / f'{args.embedding}_{ds}_{b.replace("(", "_").replace(")", "_")}_{m}')

    # def change_facet_titles(g):
    #     # Required workaround for set_titles, see https://github.com/mwaskom/seaborn/issues/509#issuecomment-316132303
    #     for row in g.axes:
    #         row[-1].texts = []
    #     return g.set_titles(row_template='{row_name}', col_template='{col_name}')
    # for dataset in np.unique(embeddings.dataset):
    #     print(dataset)
    #     embeddings_subset = embeddings[embeddings.dataset == dataset]
    #     print(embeddings_subset.shape)
    #     g = sns.relplot(x=f'{args.embedding}1', y=f'{args.embedding}2', col='task', row='method', row_order=row_order, hue='Batch', kind='scatter', data=embeddings_subset, alpha=0.5, facet_kws={'sharex': False, 'sharey': False, 'margin_titles': True}, palette='husl')
    #     g = change_facet_titles(g)
    #     plt.savefig(embeddings_folder / f'{dataset}_{args.embedding}_facetgrid_batch.png')
    #     plt.savefig(embeddings_folder / f'{dataset}_{args.embedding}_facetgrid_batch.svg')
    #     g = sns.relplot(x=f'{args.embedding}1', y=f'{args.embedding}2', col='task', row='method', row_order=row_order, hue='Cell type', kind='scatter', data=embeddings_subset, alpha=0.5, facet_kws={'sharex': False, 'sharey': False, 'margin_titles': True}, palette='Dark2')
    #     g = change_facet_titles(g)
    #     plt.savefig(embeddings_folder / f'{dataset}_{args.embedding}_facetgrid_celltype.png')
    #     plt.savefig(embeddings_folder / f'{dataset}_{args.embedding}_facetgrid_celltype.svg')





