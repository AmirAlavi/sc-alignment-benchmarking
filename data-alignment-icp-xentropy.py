# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
#    ### Imports & constants

#%%
import time
from collections import defaultdict
from pathlib import Path
from os import makedirs
from os.path import exists, join
import importlib

#get_ipython().run_line_magic('matplotlib', 'inline')
import anndata
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from IPython import display

import icp
import data
import embed
import alignment_task
import comparison_plots
import metrics
import runners
import cli
importlib.reload(icp)
importlib.reload(data)
importlib.reload(embed)
importlib.reload(alignment_task)
importlib.reload(comparison_plots)
importlib.reload(runners)
importlib.reload(cli)

#%%
parser = cli.get_parser()

# methods = ['None', 'MNN', 'SeuratV3', 'ScAlign', 'ICP', 'ICP2', 'ICP2_xentropy']
#arguments = '--methods SeuratV3 --datasets panc8 --input_space GENE --epochs=5 --no_standardize'
arguments = '--methods SeuratV3 --datasets panc8 --input_space GENE --epochs=10'
#arguments = '--methods SeuratV3 --datasets panc8-all --input_space GENE --epochs=10 --seurat_env_path C:\\Users\\Amir\\Anaconda3\\envs\\seuratV3'
args = parser.parse_args(arguments.split())

#%%
datasets = {}
alignment_tasks = []

#%%
if 'Kowalcyzk' in args.datasets:
    datasets['Kowalcyzk'] = data.get_data('Kowalcyzk')
    embed.embed(datasets, 'Kowalcyzk', args.n_PC, do_standardize=not args.no_standardize)
    embed.visualize(datasets, 'Kowalcyzk', cell_type_key='cell_type', batch_key='cell_age')
    alignment_tasks.append(alignment_task.AlignmentTask('Kowalcyzk', 'cell_age', 'cell_type', 'young', 'old'))
    alignment_tasks.append(alignment_task.AlignmentTask('Kowalcyzk', 'cell_age', 'cell_type', 'young', 'old', 'LT'))
    alignment_tasks.append(alignment_task.AlignmentTask('Kowalcyzk', 'cell_age', 'cell_type', 'young', 'old', 'MPP'))
    alignment_tasks.append(alignment_task.AlignmentTask('Kowalcyzk', 'cell_age', 'cell_type', 'young', 'old', 'ST'))

if 'CellBench' in args.datasets:
    datasets['CellBench'] = data.get_data('CellBench')
    embed.embed(datasets, 'CellBench', args.n_PC, do_standardize=not args.no_standardize)
    embed.visualize(datasets, 'CellBench', cell_type_key='cell_line_demuxlet', batch_key='protocol')
    alignment_tasks.append(alignment_task.AlignmentTask('CellBench', 'protocol', 'cell_line_demuxlet', 'Dropseq', 'CELseq2'))
    # alignment_tasks.append(alignment_task.AlignmentTask('CellBench', 'protocol', 'cell_line_demuxlet', 'Dropseq', 'CELseq2', 'H1975'))
    # alignment_tasks.append(alignment_task.AlignmentTask('CellBench', 'protocol', 'cell_line_demuxlet', 'Dropseq', 'CELseq2', 'H2228'))
    # alignment_tasks.append(alignment_task.AlignmentTask('CellBench', 'protocol', 'cell_line_demuxlet', 'Dropseq', 'CELseq2', 'HCC827'))

if 'panc8' in args.datasets:
    datasets['panc8'] = data.get_data('panc8')
    #embed.embed(datasets, 'panc8', args.n_PC, do_standardize=not args.no_standardize)
    #embed.visualize(datasets, 'panc8', cell_type_key='celltype', batch_key='dataset')
    alignment_tasks.append(alignment_task.AlignmentTask('panc8', 'dataset', 'celltype', 'celseq', 'fluidigmc1'))
    alignment_tasks.append(alignment_task.AlignmentTask('panc8', 'dataset', 'celltype', 'celseq', 'fluidigmc1', 'alpha'))
    # alignment_tasks.append(alignment_task.AlignmentTask('panc8', 'dataset', 'celltype', 'celseq', 'celseq2', 'beta'))


#%%
# Run Alignment tasks
        
print('Alignment tasks:')
for task in alignment_tasks:
    print('\t{}'.format(task))


tsne_fig, tsne_outer_grid, pca_fig, pca_outer_grid, umap_fig, umap_outer_grid, lisi_fig, lisi_outer_grid = comparison_plots.setup_comparison_grid_plot(alignment_tasks, args.methods)

import numpy as np
def plot_aligned_embedding(log_dir, adata, embed_key, alignment_task):
    plt.figure()
    plt.title('Color by batch', fontsize='small')
    batch_colors = ['m', 'c']
    for batch, color in zip(np.unique(adata.obs[alignment_task.batch_key]), batch_colors):
        idx = np.where(adata.obs[alignment_task.batch_key] == batch)[0]
        plt.scatter(adata.obsm[embed_key][idx, 0], adata.obsm[embed_key][idx, 1], c=color, label=batch, alpha=0.3)
    plt.savefig(join(log_dir, '{}_by_batch.png'.format(embed_key)))
    plt.savefig(join(log_dir, '{}_by_batch.pdf'.format(embed_key)))
    plt.savefig(join(log_dir, '{}_by_batch.svg'.format(embed_key)))
    plt.close()
    # Plot, coloring by the cell type
    plt.figure()
    plt.title('Color by cell type', fontsize='small')
    for ct in np.unique(adata.obs[alignment_task.ct_key]):
        idx = np.where(adata.obs[alignment_task.ct_key] == ct)[0]
        plt.scatter(adata.obsm[embed_key][idx, 0], adata.obsm[embed_key][idx, 1], label=ct, alpha=0.3)
    plt.savefig(join(log_dir, '{}_by_celltype.png'.format(embed_key)))
    plt.savefig(join(log_dir, '{}_by_celltype.pdf'.format(embed_key)))
    plt.savefig(join(log_dir, '{}_by_celltype.svg'.format(embed_key)))
    plt.close()

def plot_alignment_results(log_dir, adata, method_key, alignment_task):
    if method_key == 'None':
        method_key = ''
    else:
        method_key = method_key + '_'
    plot_aligned_embedding(log_dir, adata, method_key+'TSNE', alignment_task)
    plot_aligned_embedding(log_dir, adata, method_key+'PCA', alignment_task)
    plot_aligned_embedding(log_dir, adata, method_key+'UMAP', alignment_task)
    comparison_plots.plot_embedding_in_grid(adata, method_key+'TSNE', alignment_task, tsne_fig, tsne_outer_grid, i+1, j+1)
    comparison_plots.plot_embedding_in_grid(adata, method_key+'PCA', alignment_task, pca_fig, pca_outer_grid, i+1, j+1)
    comparison_plots.plot_embedding_in_grid(adata, method_key+'UMAP', alignment_task, umap_fig, umap_outer_grid, i+1, j+1)

def create_working_directory(out_path):
    if exists(out_path):
        time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
        out_path = '{}_{}'.format(out_path, time_str)
    makedirs(out_path)
    return out_path

experiment_name = 'experiment' if args.name is None else args.name
log_dir_root = create_working_directory(experiment_name)
print('Working Directory: {}\n\n'.format(log_dir_root))

# For each alignment task
for j, task in enumerate(alignment_tasks):
    print(task)
    if task.leave_out_ct is not None:
        task_idx = (datasets[task.ds_key].obs[task.batch_key] == task.source_batch) | ((datasets[task.ds_key].obs[task.batch_key] == task.target_batch) & (datasets[task.ds_key].obs[task.ct_key] != task.leave_out_ct))
    else:
        task_idx = (datasets[task.ds_key].obs[task.batch_key] == task.source_batch) | (datasets[task.ds_key].obs[task.batch_key] == task.target_batch)
    task_adata = datasets[task.ds_key][task_idx]
    
    lisi_scores = []
    
    # For each alignment method
    for i, method in enumerate(args.methods):
        print('\t{}'.format(method))
        method_key = '{}_aligned'.format(method)
        log_dir = join(log_dir_root, '{}_{}'.format(task.as_path(), method))
        if not exists(log_dir):
            makedirs(log_dir)

        if method == 'None':
            pass
            # 
            plot_alignment_results(log_dir, task_adata, method, task)
            lisi_scores.append(metrics.lisi2(task_adata.obsm['PCA'], task_adata.obs, [task.batch_key, task.ct_key], perplexity=30))
        else:
            if 'ICP' in method:
                runners.run_ICP_methods(datasets, task, task_adata, method, log_dir, args)
                #lisi_scores.append(metrics.lisi2(task_adata.obsm[method_key], task_adata.obs, [task.batch_key, task.ct_key], perplexity=30)
            elif method == 'ScAlign':
                runners.run_scAlign(datasets, task, task_adata, method, log_dir, args)
            elif method == 'MNN':
                runners.run_MNN(datasets, task, task_adata, method, log_dir, args)
            elif method == 'SeuratV3':
                #task_adata = datasets[task.ds_key]
                runners.run_Seurat(datasets, task, task_adata, method, log_dir, args)
                #runners.run_Seurat(datasets, task, datasets[task.ds_key], method, log_dir, args)
            task_adata.obsm[method_key+'_TSNE'] = TSNE(n_components=2).fit_transform(task_adata.obsm[method_key])
            task_adata.obsm[method_key+'_PCA'] = PCA(n_components=2).fit_transform(task_adata.obsm[method_key])
            task_adata.obsm[method_key+'_UMAP'] = umap.UMAP().fit_transform(task_adata.obsm[method_key])
            # plot_aligned_embedding(log_dir, task_adata, method_key+'_TSNE', task)
            # plot_aligned_embedding(log_dir, task_adata, method_key+'_PCA', task)
            # plot_aligned_embedding(log_dir, task_adata, method_key+'_UMAP', task)
            # comparison_plots.plot_embedding_in_grid(task_adata, method_key+'_TSNE', task, tsne_fig, tsne_outer_grid, i+1, j+1)
            # comparison_plots.plot_embedding_in_grid(task_adata, method_key+'_PCA', task, pca_fig, pca_outer_grid, i+1, j+1)
            # comparison_plots.plot_embedding_in_grid(task_adata, method_key+'_UMAP', task, umap_fig, umap_outer_grid, i+1, j+1)
            plot_alignment_results(log_dir, task_adata, method_key, task)
            lisi_scores.append(metrics.lisi2(task_adata.obsm[method_key], task_adata.obs, [task.batch_key, task.ct_key], perplexity=30))

    comparison_plots.plot_lisi(lisi_scores, args.methods, task, lisi_fig, lisi_outer_grid, 1, j)
tsne_fig.savefig(join(log_dir_root, 'comparison_tsne.pdf'))
tsne_fig.savefig(join(log_dir_root, 'comparison_tsne.svg'))
tsne_fig.savefig(join(log_dir_root, 'comparison_tsne.png'))
pca_fig.savefig(join(log_dir_root, 'comparison_pca.pdf'))
pca_fig.savefig(join(log_dir_root, 'comparison_pca.svg'))
pca_fig.savefig(join(log_dir_root, 'comparison_pca.png'))
lisi_fig.savefig(join(log_dir_root, 'comparison_scores.pdf'))
lisi_fig.savefig(join(log_dir_root, 'comparison_scores.svg'))
lisi_fig.savefig(join(log_dir_root, 'comparison_scores.png'))


 #%%


#%%
