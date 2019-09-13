from functools import partial
import subprocess

import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import anndata
import mnnpy
from scalign import ScAlign

import icp
import alignment_task

def run_ICP_methods(datasets, task, task_adata, method_name, log_dir):
    method_key = '{}_aligned'.format(method_name)
    A, B, type_index_dict, combined_meta = alignment_task.get_source_target(datasets, task, use_PCA=True)
    print(A.shape)
    print(B.shape)
    if method_name == 'ICP':
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
    elif method_name == 'ICP2':
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
    elif method_name == 'ICP2_xentropy':
        loss_fcn = partial(icp.relaxed_match_loss, source_match_threshold=0.5, do_mean=False)
        # aligner = icp.ICP(A, B, type_index_dict, loss_function=loss_fcn, max_iters=200, verbose=False, use_xentropy_loss=True)
        aligner = icp.ICP(A, B, type_index_dict,
                            working_dir=log_dir,
                            mse_loss_function=loss_fcn,
                            n_layers=1,
                            bias=True,
                            #act='tanh',
                            epochs=3,
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

def run_scAlign(datasets, task, task_adata, method_name, log_dir):
    method_key = '{}_aligned'.format(method_name)
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

def run_MNN(datasets, task, task_adata, method_name, log_dir):
    method_key = '{}_aligned'.format(method_name)
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

def run_Seurat(datasets, task, task_adata, method_name, log_dir):
    method_key = '{}_aligned'.format(method_name)
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