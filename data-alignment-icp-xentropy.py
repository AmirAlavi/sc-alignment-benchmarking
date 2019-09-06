# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
from IPython import get_ipython

#%% [markdown]
#    #### Docs for VS Code & Jupyter notebooks [here](https://code.visualstudio.com/docs/python/jupyter-support)
#    # Jump to sections of interest:
#    1. Visualizing Raw Datasets
#      1. [Kowalcyzk et al.](#kowal)
#      2. [CellBench](#cellbench)
#    2. Alignment Method Experiments Results
#      1. [Iterative Closest Point (ICP)](#icp)
#      2. [ICP 2](#icp2)
#      3. [ScAlign](#scalign)
#    3. [LISI Performance Metric](#lisi)
#%% [markdown]
#   ### Imports & constants

#%%
import subprocess
from collections import defaultdict
from functools import partial
from pathlib import Path
from os import makedirs
from os.path import exists, join
import importlib

#get_ipython().run_line_magic('matplotlib', 'inline')
import anndata
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from IPython import display
import torch
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
import mnnpy

from scalign import ScAlign

import icp
import data
import embed
import alignment_task
import comparison_plots
import metrics
importlib.reload(icp)
importlib.reload(data)
importlib.reload(embed)
importlib.reload(alignment_task)
importlib.reload(comparison_plots)

N_PC = 100

DO_STANDARDIZE = False

#%% [markdown]
#    # Load datasets, clean them, view reduced dimensions

#%%
datasets = {}

#%% [markdown]
## Dataset: Kowalcyzk et al.

#%%
datasets['Kowalcyzk'] = data.get_data('Kowalcyzk')
embed.embed(datasets, 'Kowalcyzk', N_PC, do_standardize=DO_STANDARDIZE)
embed.visualize(datasets, 'Kowalcyzk', cell_type_key='cell_type', batch_key='cell_age')


#%% [markdown]
## Dataset: CellBench

#%%
datasets['CellBench'] = data.get_data('CellBench')
embed.embed(datasets, 'CellBench', N_PC, do_standardize=DO_STANDARDIZE)
embed.visualize(datasets, 'CellBench', cell_type_key='cell_line_demuxlet', batch_key='protocol')

#%% [markdown]
## Dataset: panc8

#%%
datasets['panc8'] = data.get_data('panc8')
embed.embed(datasets, 'panc8', N_PC, do_standardize=DO_STANDARDIZE)
embed.visualize(datasets, 'panc8', cell_type_key='celltype', batch_key='dataset')

#%%    
# Select Alignment tasks
alignment_tasks = []
#alignment_tasks.append(alignment_task.AlignmentTask('CellBench', 'protocol', 'cell_line_demuxlet', 'Dropseq', 'CELseq2'))
#alignment_tasks.append(alignment_task.AlignmentTask('CellBench', 'protocol', 'cell_line_demuxlet', 'Dropseq', 'CELseq2', 'H1975'))
#alignment_tasks.append(alignment_task.AlignmentTask('CellBench', 'protocol', 'cell_line_demuxlet', 'Dropseq', 'CELseq2', 'H2228'))
#alignment_tasks.append(alignment_task.AlignmentTask('CellBench', 'protocol', 'cell_line_demuxlet', 'Dropseq', 'CELseq2', 'HCC827'))
alignment_tasks.append(alignment_task.AlignmentTask('Kowalcyzk', 'cell_age', 'cell_type', 'young', 'old'))
alignment_tasks.append(alignment_task.AlignmentTask('Kowalcyzk', 'cell_age', 'cell_type', 'young', 'old', 'LT'))
#alignment_tasks.append(alignment_task.AlignmentTask('Kowalcyzk', 'cell_age', 'cell_type', 'young', 'old', 'MPP'))
#alignment_tasks.append(alignment_task.AlignmentTask('Kowalcyzk', 'cell_age', 'cell_type', 'young', 'old', 'ST'))
for task in alignment_tasks:
    print(task)
# Select alignment methods:
methods = ['SeuratV3']
#methods = ['None', 'ICP', 'ICP2', 'ICP2_xentropy', 'ScAlign', 'MNN']
#methods = ['None', 'ICP', 'ICP2_xentropy']
#methods = [None, 'ScAlign']
#methods = [None, 'MNN']

tsne_fig, tsne_outer_grid, pca_fig, pca_outer_grid, lisi_fig, lisi_outer_grid = comparison_plots.setup_comparison_grid_plot(alignment_tasks, methods)

log_dir_root = 'experiments_leave_out_Kowal'


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
                                  epochs=10,
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
                                  epochs=200,
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
                                  epochs=200,
                                  lr=1e-3,
                                  momentum=0.9,
                                  l2_reg=0.,
                                  xentropy_loss_weight=10.0)
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
            cmd = "C:\\Users\\Amir\\Anaconda3\\envs\\seuratV3\\Scripts\\Rscript.exe  seurat_align.R {}".format(task.batch_key)
            print("Running command: {}".format(cmd))
            subprocess.run(cmd.split())
            aligned_adata = anndata.read_loom("_tmp_adata_for_seurat.loom")

    comparison_plots.plot_lisi(lisi_scores, methods, task, lisi_fig, lisi_outer_grid, 1, j)
tsne_fig.savefig('comparison_tsne.pdf')
tsne_fig.savefig('comparison_tsne.svg')
tsne_fig.savefig('comparison_tsne.png')
pca_fig.savefig('comparison_pca.pdf')
pca_fig.savefig('comparison_pca.svg')
pca_fig.savefig('comparison_pca.png')
lisi_fig.savefig('comparison_scores.pdf')
lisi_fig.savefig('comparison_scores.svg')
lisi_fig.savefig('comparison_scores.png')


#%%
