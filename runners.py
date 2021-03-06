# import pdb; pdb.set_trace()
import math
from functools import partial
import subprocess
from pathlib import Path
import tempfile
import platform
import pickle
import datetime

import torch
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import anndata
import mnnpy
from scalign import ScAlign
import pandas as pd

import icp
import alignment_task
from scipr import AffineSCIPR, StackedAutoEncoderSCIPR

def pretty_tdelta(tdelta):
    hours, rem = divmod(tdelta.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return '{:02d}h:{:02d}m:{:02d}s'.format(hours, mins, secs)

def run_ICP_rigid(datasets, task, task_adata, method_name, log_dir, args):
    method_key = '{}_aligned'.format(method_name)
    A, B, type_index_dict, combined_meta = alignment_task.get_source_target(datasets, task, use_PCA=args.input_space == 'PCA', subsample=args.subsample, n_subsample=args.n_subsample)

    R, t = icp.ICP_rigid(A, B, args, args.max_steps, args.tolerance, args.input_normalization)
    
    normalization = args.input_normalization
    A, B, type_index_dict, combined_meta = alignment_task.get_source_target(datasets, task, use_PCA=args.input_space == 'PCA', subsample=False)
    if normalization == 'std':
        print('Applying Standard Scaling')
        scaler = StandardScaler().fit(np.concatenate((A,B)))
        A = scaler.transform(A)
        B = scaler.transform(B)
    elif normalization == 'l2':
        print('Applying L2 Normalization')
        A = sklearn.preprocessing.normalize(A)
        B = sklearn.preprocessing.normalize(B)
    elif normalization == 'log':
        print('Applying log normalization')
        A  = np.log1p(A / A.sum(axis=1, keepdims=True) * 1e4)
        B  = np.log1p(B / B.sum(axis=1, keepdims=True) * 1e4)
    A = np.dot(R, A.T).T + t
    print(A.shape)
    n_samples = task_adata.shape[0]
    n_dims = A.shape[1]
    task_adata.obsm[method_key] = np.zeros((n_samples, n_dims))
    a_idx = np.where(task_adata.obs[task.batch_key] == task.source_batch)[0]
    b_idx = np.where(task_adata.obs[task.batch_key] == task.target_batch)[0]
    task_adata.obsm[method_key][a_idx, :] = A
    task_adata.obsm[method_key][b_idx, :] = B
    print(f'method_key: {method_key}')

def run_ICP_affine(datasets, task, task_adata, method_name, log_dir, args):
    method_key = '{}_aligned'.format(method_name)
    A, B, type_index_dict, combined_meta = alignment_task.get_source_target(datasets, task, use_PCA=args.input_space == 'PCA', subsample=args.subsample, n_subsample=args.n_subsample, leave_out_source_ct=True)

    scipr = AffineSCIPR(n_iter=args.max_steps,
                        n_epochs_per_iter=args.max_epochs,
                        matching_algo=args.matching_algo,
                        opt=args.opt,
                        lr=args.lr,
                        input_normalization=args.input_normalization,
                        frac_matches_to_keep=args.source_match_thresh,
                        source_match_thresh=args.source_match_thresh,
                        target_match_limit=args.target_match_limit)
    
    t0 = datetime.datetime.now()
    n_pairs, mean_distances = scipr.fit(A, B)
    t1 = datetime.datetime.now()
    time_str = pretty_tdelta(t1 - t0)
    print(f'took: {time_str}')
    with open(log_dir / 'fit_time.txt', 'w') as f:
        f.write(time_str + '\n')
    np.save(log_dir / 'fit_n_pairs.npy', n_pairs)
    np.save(log_dir / 'fit_mean_distances.npy', mean_distances)
    with open(log_dir / 'scipr_model.pkl', 'wb') as f:
        pickle.dump(scipr, f)
    A, B, type_index_dict, combined_meta = alignment_task.get_source_target(datasets, task, use_PCA=args.input_space == 'PCA', subsample=False, leave_out_source_ct=False)
    B = scipr.apply_input_normalization(B)
    A = scipr.transform(A)
    print(A.shape)
    n_samples = task_adata.shape[0]
    n_dims = A.shape[1]
    task_adata.obsm[method_key] = np.zeros((n_samples, n_dims))
    a_idx = np.where(task_adata.obs[task.batch_key] == task.source_batch)[0]
    b_idx = np.where(task_adata.obs[task.batch_key] == task.target_batch)[0]
    task_adata.obsm[method_key][a_idx, :] = A
    task_adata.obsm[method_key][b_idx, :] = B
    print(f'method_key: {method_key}')

def run_ICP_stacked_aes(datasets, task, task_adata, method_name, log_dir, args):
    method_key = '{}_aligned'.format(method_name)
    A, B, type_index_dict, combined_meta = alignment_task.get_source_target(datasets, task, use_PCA=args.input_space == 'PCA', subsample=args.subsample, n_subsample=args.n_subsample)

    # hidden_sizes=[256, 128, 64]
    scipr = StackedAutoEncoderSCIPR(hidden_sizes=args.stacked_aes_sizes, act=args.act, n_iter=args.max_steps,
                                    n_epochs_per_iter=args.max_epochs,
                                    matching_algo=args.matching_algo,
                                    opt=args.opt,
                                    lr=args.lr,
                                    input_normalization=args.input_normalization,
                                    frac_matches_to_keep=args.source_match_thresh,
                                    source_match_thresh=args.source_match_thresh,
                                    target_match_limit=args.target_match_limit)
    
    scipr.fit(A, B)
    with open(log_dir / 'scipr_model.pkl', 'wb') as f:
        pickle.dump(scipr, f)
    A, B, type_index_dict, combined_meta = alignment_task.get_source_target(datasets, task, use_PCA=args.input_space == 'PCA', subsample=False)
    B = scipr.apply_input_normalization(B)
    A = scipr.transform(A)
    print(A.shape)
    n_samples = task_adata.shape[0]
    n_dims = A.shape[1]
    task_adata.obsm[method_key] = np.zeros((n_samples, n_dims))
    a_idx = np.where(task_adata.obs[task.batch_key] == task.source_batch)[0]
    b_idx = np.where(task_adata.obs[task.batch_key] == task.target_batch)[0]
    task_adata.obsm[method_key][a_idx, :] = A
    task_adata.obsm[method_key][b_idx, :] = B
    print(f'method_key: {method_key}')

    
def run_ICP_methods(datasets, task, task_adata, method_name, log_dir, args):
    if 'ICP_align' in method_name:
        method_key = method_name
    else:
        method_key = '{}_aligned'.format(method_name)

        
    A, B, type_index_dict, combined_meta = alignment_task.get_source_target(datasets, task, use_PCA=args.input_space == 'PCA', subsample=args.subsample, n_subsample=args.n_subsample)
    # if args.input_space == 'PCA':
    #     A, B, type_index_dict, combined_meta = alignment_task.get_source_target(datasets, task, use_PCA=True)
    # else:
    #     A, B, type_index_dict, combined_meta = alignment_task.get_source_target(datasets, task, use_PCA=False)
    print(A.shape)
    print(B.shape)
    if method_name == 'ICP':
        #aligner = icp.ICP(A, B, type_index_dict, loss_function=icp.closest_point_loss, max_iters=200, verbose=False)
        aligner = icp.ICP(A, B, type_index_dict,
                            working_dir=log_dir,
                            mse_loss_function=icp.closest_point_loss,
                            n_layers=args.nlayers,
                            bias=args.bias,
                            act=args.act,
                            epochs=args.epochs,
                            lr=args.lr,
                            momentum=0.9,
                            l2_reg=args.l2_reg,
                            xentropy_loss_weight=0.0,
                            plot_every_n_steps=args.plot_every_n)
    elif method_name == 'ICP2':
        loss_fcn = partial(icp.relaxed_match_loss, source_match_threshold=args.source_match_thresh, do_mean=False)
        #aligner = icp.ICP(A, B, type_index_dict, loss_function=loss_fcn, max_iters=200, verbose=False)
        aligner = icp.ICP(A, B, type_index_dict,
                            working_dir=log_dir,
                            mse_loss_function=loss_fcn,
                            n_layers=args.nlayers,
                            bias=args.bias,
                            act=args.act,
                            epochs=args.epochs,
                            lr=args.lr,
                            momentum=0.9,
                            l2_reg=args.l2_reg,
                            xentropy_loss_weight=0.0,
                            plot_every_n_steps=args.plot_every_n)
#             elif method == 'ICP2_act':
#                 loss_fcn = partial(icp.relaxed_match_loss, source_match_threshold=0.5)
#                 aligner = icp.ICP(A, B, type_index_dict, act='tanh', loss_function=loss_fcn, max_iters=200, verbose=False)
#             elif method == 'ICP2_act+lin':
#                 loss_fcn = partial(icp.relaxed_match_loss, source_match_threshold=0.5)
#                 aligner = icp.ICP(A, B, type_index_dict, n_layers=2, act='tanh', loss_function=loss_fcn, max_iters=200, verbose=False)
    elif method_name == 'ICP2_xentropy':
        loss_fcn = partial(icp.relaxed_match_loss, source_match_threshold=args.source_match_thresh, do_mean=False)
        # aligner = icp.ICP(A, B, type_index_dict, loss_function=loss_fcn, max_iters=200, verbose=False, use_xentropy_loss=True)
        aligner = icp.ICP(A, B, type_index_dict,
                            working_dir=log_dir,
                            mse_loss_function=loss_fcn,
                            n_layers=args.nlayers,
                            bias=args.bias,
                            act=args.act,
                            epochs=args.max_steps,
                            lr=args.lr,
                            momentum=0.9,
                            l2_reg=args.l2_reg,
                            xentropy_loss_weight=args.xentropy_loss_wt,
                            plot_every_n_steps=args.plot_every_n)
    elif 'ICP_align' in method_name:
        if args.matching_algo == 'closest':
            assignment_fn = icp.assign_closest_points
            print('USING MATCHING ALGO: CLOSEST')
        elif args.matching_algo == 'greedy':
            assignment_fn = partial(icp.assign_greedy, source_match_threshold=args.source_match_thresh, target_match_limit=args.target_match_limit)
            print('USING MATCHING ALGO: GREEDY')
        elif args.matching_algo == 'hungarian':
            n_to_match = math.floor(args.source_match_thresh * min(A.shape[0], B.shape[0]))
            assignment_fn = partial(icp.assign_hungarian, n_to_match=n_to_match)
            print('USING MATCHING ALGO: HUNGARIAN')
        if args.input_space == 'GENE' and args.enforce_pos:
            enforce_pos=True
        else:
            enforce_pos=False
        aligner = icp.ICP_converge(A, B, type_index_dict,
                                   working_dir=log_dir,
                                   assignment_fn=assignment_fn,
                                   enforce_pos=enforce_pos,
                                   n_layers=args.nlayers,
                                   bias=args.bias,
                                   act=args.act,
                                   min_steps=args.min_steps,
                                   max_steps=args.max_steps,
                                   tolerance=args.tolerance,
                                   patience=args.patience,
                                   max_epochs=args.max_epochs,
                                   lr=args.lr,
                                   momentum=0.9,
                                   l2_reg=args.l2_reg,
                                   xentropy_loss_weight=args.xentropy_loss_wt,
                                   plot_every_n_steps=args.plot_every_n,
                                   mini_batching=args.mini_batching,
                                   batch_size=args.batch_size,
                                   normalization=args.input_normalization,
                                   use_autoencoder=args.use_autoencoder,
                                   last_layer_linear=args.last_layer_linear,
                                   dropout=args.dropout,
                                   batch_norm=args.batch_norm,
                                   sparse_training=args.sparse,
                                   cpu_only=args.cpu_only,
                                   optimizer=args.opt)
    elif method_name == 'ICP2_xentropy_converge':
        assignment_fn = partial(icp.assign_greedy, source_match_threshold=args.source_match_thresh)
        aligner = icp.ICP_converge(A, B, type_index_dict,
                                   working_dir=log_dir,
                                   assignment_fn=assignment_fn,
                                   n_layers=args.nlayers,
                                   bias=args.bias,
                                   act=args.act,
                                   steps=args.steps,
                                   max_epochs=args.epochs,
                                   lr=args.lr,
                                   momentum=0.9,
                                   l2_reg=args.l2_reg,
                                   xentropy_loss_weight=args.xentropy_loss_wt,
                                   plot_every_n_steps=args.plot_every_n)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aligner_fcn = lambda x: aligner(torch.from_numpy(x).float().to(device)).detach().cpu().numpy()
    #standardizing because it was fitted with standardized data (see ICP code)
    normalization = args.input_normalization
    A, B, type_index_dict, combined_meta = alignment_task.get_source_target(datasets, task, use_PCA=args.input_space == 'PCA', subsample=False)
    if normalization == 'std':
        print('Applying Standard Scaling')
        scaler = StandardScaler().fit(np.concatenate((A,B)))
        A = scaler.transform(A)
        B = scaler.transform(B)
    elif normalization == 'l2':
        print('Applying L2 Normalization')
        A = sklearn.preprocessing.normalize(A)
        B = sklearn.preprocessing.normalize(B)
    elif normalization == 'log':
        print('Applying log normalization')
        A  = np.log1p(A / A.sum(axis=1, keepdims=True) * 1e4)
        B  = np.log1p(B / B.sum(axis=1, keepdims=True) * 1e4)
    A = aligner_fcn(A)
    print(A.shape)
    n_samples = task_adata.shape[0]
    n_dims = A.shape[1]
    task_adata.obsm[method_key] = np.zeros((n_samples, n_dims))
    a_idx = np.where(task_adata.obs[task.batch_key] == task.source_batch)[0]
    b_idx = np.where(task_adata.obs[task.batch_key] == task.target_batch)[0]
    task_adata.obsm[method_key][a_idx, :] = A
    task_adata.obsm[method_key][b_idx, :] = B
    print(f'method_key: {method_key}')

def run_scAlign(datasets, task, task_adata, method_name, log_dir, args):
    method_key = '{}_aligned'.format(method_name)
    #idx = (datasets['CellBench'].obs['cell_line_demuxlet'] == 'H2228') & (datasets['CellBench'].obs['protocol'] == 'CELseq2')
            #datasets['CellBench'] = datasets['CellBench'][ ~idx ,:]
    sc_align = ScAlign(
        object1_name=task.source_batch,
        object2_name=task.target_batch, 
        object_var=task.batch_key,
        label_var=task.ct_key,
        data_use=args.input_space,
        user_options={
            'max_steps': args.scalign_max_steps,
            'batch_size': args.scalign_batch_size,
            'learning_rate': args.scalign_lr,
            'architecture': args.scalign_architecture,
            'emb_size': args.scalign_emb_size,
            'logdir': 'scAlign_model',
            'log_results': True,
            'early_stop': True
        },
        device='CPU')
    t0 = datetime.datetime.now()
    sc_align.fit_encoder(task_adata)
    t1 = datetime.datetime.now()
    time_str = pretty_tdelta(t1 - t0)
    print(f'took: {time_str}')
    with open(log_dir / 'fit_time.txt', 'w') as f:
        f.write(time_str + '\n')
    print('Trained encoder saved to: {}'.format(sc_align.trained_encoder_path_))
    if args.input_space == 'PCA':
        data_to_encode = task_adata.obsm['PCA']
    else:
        data_to_encode = task_adata.X
    task_adata.obsm[method_key] = sc_align.encode(data_to_encode)

def run_MNN(datasets, task, task_adata, method_name, log_dir, args):
    method_key = '{}_aligned'.format(method_name)
    A_idx = task_adata.obs[task.batch_key] == task.source_batch
    B_idx = task_adata.obs[task.batch_key] == task.target_batch
    if args.input_space == 'PCA':
        A_X = task_adata[A_idx].obsm['PCA']
        B_X = task_adata[B_idx].obsm['PCA']
    else:
        A_X = task_adata[A_idx].X
        B_X = task_adata[B_idx].X
#             # standardizing
#             scaler = StandardScaler().fit(np.concatenate((A_X,B_X)))
#             A_X = scaler.transform(A_X)
#             B_X = scaler.transform(B_X)
    mnn_adata_A = anndata.AnnData(X=A_X, obs=task_adata[A_idx].obs)
    mnn_adata_B = anndata.AnnData(X=B_X, obs=task_adata[B_idx].obs)
    t0 = datetime.datetime.now()
    corrected = mnnpy.mnn_correct(mnn_adata_A, mnn_adata_B)
    t1 = datetime.datetime.now()
    time_str = pretty_tdelta(t1 - t0)
    print(f'took: {time_str}')
    with open(log_dir / 'fit_time.txt', 'w') as f:
        f.write(time_str + '\n')
    task_adata.obsm[method_key] = np.zeros(corrected[0].shape)
    task_adata.obsm[method_key][np.where(A_idx)[0]] = corrected[0].X[:mnn_adata_A.shape[0]]
    task_adata.obsm[method_key][np.where(B_idx)[0]] = corrected[0].X[mnn_adata_A.shape[0]:]

def run_Seurat(datasets, task, task_adata, method_name, log_dir, args):
    method_key = '{}_aligned'.format(method_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        working_dir = Path(tmp_dir)
        print("saving data for Seurat")
        #task_adata.write('_tmp_adata_for_seurat.h5ad')
        if args.input_space == 'PCA':
            df = pd.DataFrame(task_adata.obsm['PCA'], index=task_adata.obs.index)
        else:
            df = task_adata.to_df()
        print(df.shape)
        #print(df.index)
        #print(df.columns)
        count_file = working_dir / '_tmp_counts.csv'
        df.T.to_csv(count_file)
        metadata_file = working_dir / '_tmp_meta.csv'
        task_adata.obs.to_csv(metadata_file)
        loom_result_file = working_dir / '_tmp_adata_for_seurat.loom'
        # Run seurat
        #cmd = "C:\\Users\\samir\\Anaconda3\\envs\\seuratV3\\Scripts\\Rscript.exe  seurat_align.R {}".format(task.batch_key)
        seurat_env_path = Path(args.seurat_env_path)
        if platform.system() == 'Windows':
            bin_path = seurat_env_path / 'Library' / 'mingw-w64' / 'bin'
            rscript_path = seurat_env_path / 'Scripts' / 'Rscript.exe'
            cmd = 'set PATH={};%PATH% && {} seurat_align.R {} {} {} {} {}'.format(bin_path, rscript_path, task.batch_key, args.seurat_dims, count_file, metadata_file, loom_result_file)
            cmd = cmd.split()
        else:
            bin_path = seurat_env_path / 'bin' 
            rscript_path = bin_path / 'Rscript'
            cmd = 'PATH="{}:$PATH" {} seurat_align.R {} {} {} {} {}'.format(bin_path, rscript_path, task.batch_key, args.seurat_dims, count_file, metadata_file, loom_result_file)
            #cmd = '{} seurat_align.R {} {} {} {} {}'.format(rscript_path, task.batch_key, args.seurat_dims, count_file, metadata_file, loom_result_file)

        #cmd = r"set PATH=C:\Users\samir\Anaconda3\envs\seuratV3\Library\mingw-w64\bin;%PATH% && C:\Users\samir\Anaconda3\envs\seuratV3\Scripts\Rscript.exe  seurat_align.R {}".format(task.batch_key)
        print('Running command: {}'.format(cmd))
        try:
            t0 = datetime.datetime.now()
            console_output = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            t1 = datetime.datetime.now()
            time_str = pretty_tdelta(t1 - t0)
            print(f'took: {time_str}')
            with open(log_dir / 'fit_time.txt', 'w') as f:
                f.write(time_str + '\n')
            console_output = console_output.stdout.decode('UTF-8')
            print('Finished running')
            print(console_output)
            aligned_adata = anndata.read_loom(loom_result_file)
            print('done loading loom')
            print(aligned_adata.shape)
            #print(type(aligned_adata.X))
            # print(aligned_adata.obs.columns)
            # print(aligned_adata.obsm.keys())
            # print('todense...')
            task_adata.obsm[method_key] = aligned_adata.X.todense()
            # print(task_adata.obsm[method_key][:5, :])
        except subprocess.CalledProcessError as e:
            print("RUNNING SEURAT FAILED")
            print(e.stdout.decode('UTF-8'))
