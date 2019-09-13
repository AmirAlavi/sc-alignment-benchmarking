# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

 
#%% [markdown]
#     #### Docs for VS Code & Jupyter notebooks [here](https://code.visualstudio.com/docs/python/jupyter-support)
#     # Jump to sections of interest:
#     1. Visualizing Raw Datasets
#       1. [Kowalcyzk et al.](#kowal)
#       2. [CellBench](#cellbench)
#     2. Alignment Method Experiments Results
#       1. [Iterative Closest Point (ICP)](#icp)
#       2. [ICP 2](#icp2)
#       3. [ScAlign](#scalign)
#     3. [LISI Performance Metric](#lisi)
#%% [markdown]
#    ### Imports & constants

#%%
import time
from collections import defaultdict
from pathlib import Path
from os import makedirs
from os.path import exists, join
import importlib

get_ipython().run_line_magic('matplotlib', 'inline')
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
importlib.reload(icp)
importlib.reload(data)
importlib.reload(embed)
importlib.reload(alignment_task)
importlib.reload(comparison_plots)
importlib.reload(runners)

N_PC = 100

DO_STANDARDIZE = False

#%% [markdown]
# # Select alignment methods to run
#%%
# Select alignment methods:
# methods = ['None', 'MNN', 'SeuratV3', 'ScAlign', 'ICP', 'ICP2', 'ICP2_xentropy']
# methods = ['ICP2_xentropy']
# methods = ['SeuratV3']
# methods = ['None', 'ICP', 'ICP2', 'ICP2_xentropy', 'SeuratV3']
# methods = ['None', 'ICP', 'ICP2', 'SeuratV3', 'ScAlign', 'MNN']
# methods = ['None', 'ICP', 'ICP2_xentropy']
# methods = ['None', 'ScAlign']
methods = ['SeuratV3']

#%% [markdown]
# # Select datasets to run all methods against

#%%
# selected_data = ['Kowalcyzk', 'CellBench', 'panc8']
selected_data = ['CellBench']
datasets = {}
alignment_tasks = []

#%% [markdown]
# # Dataset: Kowalcyzk et al.

#%%
if 'Kowalcyzk' in selected_data:
    datasets['Kowalcyzk'] = data.get_data('Kowalcyzk')
    embed.embed(datasets, 'Kowalcyzk', N_PC, do_standardize=DO_STANDARDIZE)
    embed.visualize(datasets, 'Kowalcyzk', cell_type_key='cell_type', batch_key='cell_age')
    alignment_tasks.append(alignment_task.AlignmentTask('Kowalcyzk', 'cell_age', 'cell_type', 'young', 'old'))
    alignment_tasks.append(alignment_task.AlignmentTask('Kowalcyzk', 'cell_age', 'cell_type', 'young', 'old', 'LT'))
    alignment_tasks.append(alignment_task.AlignmentTask('Kowalcyzk', 'cell_age', 'cell_type', 'young', 'old', 'MPP'))
    alignment_tasks.append(alignment_task.AlignmentTask('Kowalcyzk', 'cell_age', 'cell_type', 'young', 'old', 'ST'))

if 'CellBench' in selected_data:
    datasets['CellBench'] = data.get_data('CellBench')
    embed.embed(datasets, 'CellBench', N_PC, do_standardize=DO_STANDARDIZE)
    embed.visualize(datasets, 'CellBench', cell_type_key='cell_line_demuxlet', batch_key='protocol')
    alignment_tasks.append(alignment_task.AlignmentTask('CellBench', 'protocol', 'cell_line_demuxlet', 'Dropseq', 'CELseq2'))
    # alignment_tasks.append(alignment_task.AlignmentTask('CellBench', 'protocol', 'cell_line_demuxlet', 'Dropseq', 'CELseq2', 'H1975'))
    # alignment_tasks.append(alignment_task.AlignmentTask('CellBench', 'protocol', 'cell_line_demuxlet', 'Dropseq', 'CELseq2', 'H2228'))
    # alignment_tasks.append(alignment_task.AlignmentTask('CellBench', 'protocol', 'cell_line_demuxlet', 'Dropseq', 'CELseq2', 'HCC827'))

if 'panc8' in selected_data:
    datasets['panc8'] = data.get_data('panc8')
    embed.embed(datasets, 'panc8', N_PC, do_standardize=DO_STANDARDIZE)
    embed.visualize(datasets, 'panc8', cell_type_key='celltype', batch_key='dataset')
    alignment_tasks.append(alignment_task.AlignmentTask('panc8', 'dataset', 'celltype', 'celseq', 'celseq2'))
    alignment_tasks.append(alignment_task.AlignmentTask('panc8', 'dataset', 'celltype', 'celseq', 'celseq2', 'alpha'))
    alignment_tasks.append(alignment_task.AlignmentTask('panc8', 'dataset', 'celltype', 'celseq', 'celseq2', 'beta'))


#%%
# Run Alignment tasks
print('Alignment tasks:')
for task in alignment_tasks:
    print('\t{}'.format(task))


tsne_fig, tsne_outer_grid, pca_fig, pca_outer_grid, umap_fig, umap_outer_grid, lisi_fig, lisi_outer_grid = comparison_plots.setup_comparison_grid_plot(alignment_tasks, methods)

def create_working_directory(out_path):
    if exists(out_path):
        time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
        out_path = '{}_{}'.format(out_path, time_str)
    makedirs(out_path)
    return out_path

log_dir_root = create_working_directory('experiments')
print('Working Directory: {}\n\n'.format(log_dir_root))

#%%
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
    for i, method in enumerate(methods):
        print('\t{}'.format(method))
        method_key = '{}_aligned'.format(method)
        
        if method == 'None':
            comparison_plots.plot_embedding_in_grid(task_adata, 'TSNE', task, tsne_fig, tsne_outer_grid, i+1, j+1)
            comparison_plots.plot_embedding_in_grid(task_adata, 'PCA', task, pca_fig, pca_outer_grid, i+1, j+1)
            lisi_scores.append(metrics.lisi2(task_adata.obsm['PCA'], task_adata.obs, [task.batch_key, task.ct_key], perplexity=30))
        elif method == 'ICP' or method == 'ICP2' or method=='ICP2_act' or method == 'ICP2_act+lin' or method == 'ICP2_xentropy':
            log_dir = join(log_dir_root, '{}_{}'.format(task.as_path(), method))
            if not exists(log_dir):
                makedirs(log_dir)
            A, B, type_index_dict, combined_meta = alignment_task.get_source_target(datasets, task, use_PCA=True)
            print(A.shape)
            print(B.shape)
            if method == 'ICP':
                #aligner = icp.ICP(A, B, type_index_dict, loss_function=icp.closest_point_loss, max_iters=200, verbose=False)
                aligner = icp.ICP(A, B, type_index_dict,
                                  working_dir=log_dir,
                                  mse_loss_function=icp.closest_point_loss,
                                  n_layers=1,
                                  bias=True,
                                  #act='tanh',
                                  epochs=15,
                                  lr=1e-3,
                                  momentum=0.9,
                                  l2_reg=0.,
                                  xentropy_loss_weight=0.0)
            elif method == 'ICP2':
                loss_fcn = partial(icp.relaxed_match_loss, source_match_threshold=0.5, do_mean=False)
                #aligner = icp.ICP(A, B, type_index_dict, loss_function=loss_fcn, max_iters=200, verbose=False)
                aligner = icp.ICP(A, B, type_index_dict,
                                  working_dir=log_dir,
                                  mse_loss_function=loss_fcn,
                                  n_layers=1,
                                  bias=True,
                                  #act='tanh',
                                  epochs=15,
                                  lr=1e-3,
                                  momentum=0.9,
                                  l2_reg=0.,
                                  xentropy_loss_weight=0.0)
#             elif method == 'ICP2_act':
#                 loss_fcn = partial(icp.relaxed_match_loss, source_match_threshold=0.5)
#                 aligner = icp.ICP(A, B, type_index_dict, act='tanh', loss_function=loss_fcn, max_iters=200, verbose=False)
#             elif method == 'ICP2_act+lin':
#                 loss_fcn = partial(icp.relaxed_match_loss, source_match_threshold=0.5)
#                 aligner = icp.ICP(A, B, type_index_dict, n_layers=2, act='tanh', loss_function=loss_fcn, max_iters=200, verbose=False)
            elif method == 'ICP2_xentropy':
                loss_fcn = partial(icp.relaxed_match_loss, source_match_threshold=0.5, do_mean=False)
                # aligner = icp.ICP(A, B, type_index_dict, loss_function=loss_fcn, max_iters=200, verbose=False, use_xentropy_loss=True)
                aligner = icp.ICP(A, B, type_index_dict,
                                  working_dir=log_dir,
                                  mse_loss_function=loss_fcn,
                                  n_layers=1,
                                  bias=True,
                                  #act='tanh',
                                  epochs=15,
                                  lr=1e-3,
                                  momentum=0.9,
                                  l2_reg=0.,
                                  xentropy_loss_weight=10.0,
                                  plot_every_n_steps=5)
            aligner_fcn = lambda x: aligner(torch.from_numpy(x).float()).detach().numpy()
            #standardizing because it was fitted with standardized data (see ICP code)
            scaler = StandardScaler().fit(np.concatenate((A,B)))
            A = scaler.transform(A)
            B = scaler.transform(B)
            A = aligner_fcn(A)
            print(A.shape)
            n_samples = task_adata.shape[0]
            n_dims = A.shape[1]
            task_adata.obsm[method_key] = np.zeros((n_samples, n_dims))
            a_idx = np.where(task_adata.obs[task.batch_key] == task.source_batch)[0]
            b_idx = np.where(task_adata.obs[task.batch_key] == task.target_batch)[0]
            task_adata.obsm[method_key][a_idx, :] = A
            task_adata.obsm[method_key][b_idx, :] = B
            #lisi_scores.append(metrics.lisi2(task_adata.obsm[method_key], task_adata.obs, [task.batch_key, task.ct_key], perplexity=30)
            task_adata.obsm[method_key+'_TSNE'] = TSNE(n_components=2).fit_transform(task_adata.obsm[method_key])
            task_adata.obsm[method_key+'_PCA'] = PCA(n_components=2).fit_transform(task_adata.obsm[method_key])
            comparison_plots.plot_embedding_in_grid(task_adata, method_key+'_TSNE', task, tsne_fig, tsne_outer_grid, i+1, j+1)
            comparison_plots.plot_embedding_in_grid(task_adata, method_key+'_PCA', task, pca_fig, pca_outer_grid, i+1, j+1)
            lisi_scores.append(metrics.lisi2(task_adata.obsm[method_key], task_adata.obs, [task.batch_key, task.ct_key], perplexity=30))
        elif method == 'ScAlign':
            #idx = (datasets['CellBench'].obs['cell_line_demuxlet'] == 'H2228') & (datasets['CellBench'].obs['protocol'] == 'CELseq2')
            #datasets['CellBench'] = datasets['CellBench'][ ~idx ,:]
            sc_align = ScAlign(
                object1_name=task.source_batch,
                object2_name=task.target_batch, 
                object_var=task.batch_key,
                label_var=task.ct_key,
                data_use='PCA',
                user_options={
                    #'max_steps': 100,
                    'logdir': 'scAlign_model',
                    'log_results': True,
                    'early_stop': True
                },
                device='CPU')
            sc_align.fit_encoder(task_adata)
            print('Trained encoder saved to: {}'.format(sc_align.trained_encoder_path_))
            task_adata.obsm[method_key] = sc_align.encode(task_adata.obsm['PCA'])
            task_adata.obsm[method_key+'_TSNE'] = TSNE(n_components=2).fit_transform(task_adata.obsm[method_key])
            task_adata.obsm[method_key+'_PCA'] = PCA(n_components=2).fit_transform(task_adata.obsm[method_key])
            comparison_plots.plot_embedding_in_grid(task_adata, method_key+'_PCA', task, pca_fig, pca_outer_grid, i+1, j+1)
            comparison_plots.plot_embedding_in_grid(task_adata, method_key+'_TSNE', task, tsne_fig, tsne_outer_grid, i+1, j+1)
            lisi_scores.append(metrics.lisi2(task_adata.obsm[method_key], task_adata.obs, [task.batch_key, task.ct_key], perplexity=30))
        elif method == 'MNN':
            A_idx = task_adata.obs[task.batch_key] == task.source_batch
            B_idx = task_adata.obs[task.batch_key] == task.target_batch
            A_X = task_adata[A_idx].obsm['PCA']
            B_X = task_adata[B_idx].obsm['PCA']
#             # standardizing
#             scaler = StandardScaler().fit(np.concatenate((A_X,B_X)))
#             A_X = scaler.transform(A_X)
#             B_X = scaler.transform(B_X)
            mnn_adata_A = anndata.AnnData(X=A_X, obs=task_adata[A_idx].obs)
            mnn_adata_B = anndata.AnnData(X=B_X, obs=task_adata[B_idx].obs)
            corrected = mnnpy.mnn_correct(mnn_adata_A, mnn_adata_B)
            task_adata.obsm[method_key] = np.zeros(corrected[0].shape)
            task_adata.obsm[method_key][np.where(A_idx)[0]] = corrected[0].X[:mnn_adata_A.shape[0]]
            task_adata.obsm[method_key][np.where(B_idx)[0]] = corrected[0].X[mnn_adata_A.shape[0]:]
            task_adata.obsm[method_key+'_TSNE'] = TSNE(n_components=2).fit_transform(task_adata.obsm[method_key])
            task_adata.obsm[method_key+'_PCA'] = PCA(n_components=2).fit_transform(task_adata.obsm[method_key])
            comparison_plots.plot_embedding_in_grid(task_adata, method_key+'_PCA', task, pca_fig, pca_outer_grid, i+1, j+1)
            comparison_plots.plot_embedding_in_grid(task_adata, method_key+'_TSNE', task, tsne_fig, tsne_outer_grid, i+1, j+1)
            lisi_scores.append(metrics.lisi2(task_adata.obsm[method_key], task_adata.obs, [task.batch_key, task.ct_key], perplexity=30))
        elif method == 'SeuratV3':
            print("saving data for Seurat")
            #task_adata.write('_tmp_adata_for_seurat.h5ad')
            df = task_adata.to_df()
            df.T.to_csv('_tmp_counts.csv')
            task_adata.obs.to_csv('_tmp_meta.csv')
            # Run seurat
            #cmd = "C:\\Users\\samir\\Anaconda3\\envs\\seuratV3\\Scripts\\Rscript.exe  seurat_align.R {}".format(task.batch_key)
            cmd = r"set PATH=C:\Users\Amir\Anaconda3\envs\seuratV3\Library\mingw-w64\bin;%PATH% && C:\Users\Amir\Anaconda3\envs\seuratV3\Scripts\Rscript.exe  seurat_align.R {}".format(task.batch_key)
            print("Running command: {}".format(cmd))
            console_output = subprocess.run(cmd.split(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            console_output = console_output.stdout.decode('UTF-8')
            print(console_output)
            aligned_adata = anndata.read_loom("_tmp_adata_for_seurat.loom")
            print('done loading loom')
            #print(type(aligned_adata.X))
            print('todense...')
            task_adata.obsm[method_key] = aligned_adata.X.todense()
            print('tsne')
            task_adata.obsm[method_key+'_TSNE'] = TSNE(n_components=2).fit_transform(task_adata.obsm[method_key])
            print('pca')
            task_adata.obsm[method_key+'_PCA'] = PCA(n_components=2).fit_transform(task_adata.obsm[method_key])
            print('umap')
            umapped = umap.UMAP().fit_transform(datasets[key].X)
            plt.figure()
            plt.scatter(umapped[:, 0], umapped[:, 1])
            comparison_plots.plot_embedding_in_grid(task_adata, method_key+'_PCA', task, pca_fig, pca_outer_grid, i+1, j+1)
            comparison_plots.plot_embedding_in_grid(task_adata, method_key+'_TSNE', task, tsne_fig, tsne_outer_grid, i+1, j+1)
            print('compute lisi...')
            lisi_scores.append(metrics.lisi2(task_adata.obsm[method_key], task_adata.obs, [task.batch_key, task.ct_key], perplexity=30))

    comparison_plots.plot_lisi(lisi_scores, methods, task, lisi_fig, lisi_outer_grid, 1, j)
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


# New loop:

#tsne_fig, tsne_outer_grid, pca_fig, pca_outer_grid, umap_fig, umap_outer_grid, lisi_fig, lisi_outer_grid = comparison_plots.setup_comparison_grid_plot(alignment_tasks, methods)
# def run_method_on_task(method_name, task_adata, tsne_fig, tsne_outer_grid, pca_fig, pca_outer_grid, umap_fig, umap_outer_grid, log_dir_root):
#     log_dir = join(log_dir_root, '{}_{}'.format(task.as_path(), method))
#     if not exists(log_dir):
#         makedirs(log_dir)
#     if 'ICP' in method_name:
        
print('Alignment tasks:')
for task in alignment_tasks:
    print('\t{}'.format(task))


tsne_fig, tsne_outer_grid, pca_fig, pca_outer_grid, umap_fig, umap_outer_grid, lisi_fig, lisi_outer_grid = comparison_plots.setup_comparison_grid_plot(alignment_tasks, methods)

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

log_dir_root = create_working_directory('experiments')
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
    for i, method in enumerate(methods):
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
                runners.run_ICP_methods(datasets, task, task_adata, method, log_dir)
                #lisi_scores.append(metrics.lisi2(task_adata.obsm[method_key], task_adata.obs, [task.batch_key, task.ct_key], perplexity=30)
            elif method == 'ScAlign':
                runners.run_scAlign(datasets, task, task_adata, method, log_dir)
            elif method == 'MNN':
                runners.run_MNN(datasets, task, task_adata, method, log_dir)
            elif method == 'SeuratV3':
                runners.run_Seurat(datasets, task, task_adata, method, log_dir)
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

    comparison_plots.plot_lisi(lisi_scores, methods, task, lisi_fig, lisi_outer_grid, 1, j)
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
