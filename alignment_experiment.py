# import pdb; pdb.set_trace()
# To add a new cell, type '#%%'

# To add a new markdown cell, type '#%% [markdown]'
#%%
#from IPython import get_ipython

#%% [markdown]
#    ### Imports & constants
import sys
import pickle
import time
from collections import defaultdict
from pathlib import Path
from os import makedirs
from os.path import exists, join
import tempfile
import warnings
warnings.filterwarnings('ignore')

#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import anndata
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['svg.fonttype'] = 'none'
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
#from IPython import display

import icp
import data
from dataset_info import batch_columns, celltype_columns, batches_available, celltypes_available
import embed
import alignment_task
import comparison_plots
import metrics
import runners
import cli
from de_test import de_comparison
# import importlib
# importlib.reload(icp)
# importlib.reload(data)
# importlib.reload(embed)
# importlib.reload(alignment_task)
# importlib.reload(comparison_plots)
# importlib.reload(runners)
# importlib.reload(cli)


if __name__ == '__main__':
    parser = cli.get_parser()

    # example tasks:
    # young --> old
    # Dropseq --> CELseq2
    # celseq --> fluidigmc1

    # methods = ['None', 'MNN', 'SeuratV3', 'ScAlign', 'ICP', 'ICP2', 'ICP2_xentropy']
    #arguments = '--methods SeuratV3 --datasets panc8 --input_space GENE --epochs=5 --no_standardize'
    #arguments = '--method None --dataset panc8 --source celseq --target fluidigmc1 --input_space GENE --seurat_env_path C:\\Users\\samir\\Anaconda3\\envs\\seuratV3'
    #arguments = '--methods SeuratV3 --datasets panc8-all --input_space GENE --epochs=10 --seurat_env_path C:\\Users\\Amir\\Anaconda3\\envs\\seuratV3'
    #args = parser.parse_args(arguments.split())
    args = parser.parse_args()

    def create_working_directory(out_path):
        try:
            makedirs(out_path)
        except FileExistsError:
            time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
            out_path = tempfile.mkdtemp(prefix='{}_{}_'.format(out_path, time_str), dir='.')
        return out_path

    log_dir = create_working_directory(args.output_folder)
    log_dir = Path(log_dir)
    print('Working Directory: {}\n\n'.format(log_dir))

    for argument, value in vars(args).items():
        print('{}: {}'.format(argument, value))

    with open(log_dir / 'args.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    #%%
    datasets = {}
    datasets[args.dataset] = data.get_data(args.dataset, args)
    # if args.input_space == 'PCA' or args.method == 'None':
    embed.embed(datasets, args.dataset, args.n_PC, do_standardize=args.standardize, log_dir=log_dir)
    embed.visualize(datasets, args.dataset, cell_type_key=celltype_columns[args.dataset], batch_key=batch_columns[args.dataset], log_dir=log_dir)

    #%%
    task = alignment_task.AlignmentTask(args.dataset, batch_columns[args.dataset], celltype_columns[args.dataset], args.source, args.target, args.leaveOut)

    #%%
    # Run Alignment tasks
            
    print('Alignment task: {}'.format(task))

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



    if task.leave_out_ct is not None:
        task_idx = (datasets[task.ds_key].obs[task.batch_key] == task.source_batch) | ((datasets[task.ds_key].obs[task.batch_key] == task.target_batch) & (datasets[task.ds_key].obs[task.ct_key] != task.leave_out_ct))
    else:
        task_idx = (datasets[task.ds_key].obs[task.batch_key] == task.source_batch) | (datasets[task.ds_key].obs[task.batch_key] == task.target_batch)
    task_adata = datasets[task.ds_key][task_idx]
    method_key = '{}_aligned'.format(args.method)

    kbet_stats = None
    if args.method == 'None':
        plot_alignment_results(log_dir, task_adata, args.method, task)
        if args.input_space == 'PCA':
            lisi_score = metrics.lisi2(task_adata.obsm['PCA'], task_adata.obs, [task.batch_key, task.ct_key], perplexity=30)
            lisi_score_batch = metrics.lisi2(task_adata.obsm['PCA'], task_adata.obs, [task.batch_key], perplexity=30)
            batch_A_adata = task_adata[task_adata.obs[task.batch_key] == task.source_batch, :]
            print(f'batch_A_adata.obsm["PCA"].shape: {batch_A_adata.obsm["PCA"].shape}')
            lisi_score_celltype = metrics.lisi2(batch_A_adata.obsm['PCA'], batch_A_adata.obs, [task.ct_key], perplexity=30)
        else:
            lisi_score = metrics.lisi2(task_adata.X, task_adata.obs, [task.batch_key, task.ct_key], perplexity=30)
            lisi_score_batch = metrics.lisi2(task_adata.X, task_adata.obs, [task.batch_key], perplexity=30)
            batch_A_adata = task_adata[task_adata.obs[task.batch_key] == task.source_batch, :]
            print(f'batch_A_adata.shape: {batch_A_adata.shape}')
            lisi_score_celltype = metrics.lisi2(batch_A_adata.X, batch_A_adata.obs, [task.ct_key], perplexity=30)
        if args.do_kBET_test:
            try:
                kbet_stats = metrics.kBET(task_adata.X, task_adata.obs, task.batch_key, args.kBET_env_path)
                print(kbet_stats)
                print('kBET medians:')
                print(kbet_stats.median(axis=0))
            except Exception:
                print('kBET failed')
    else:
        if args.method == 'ICP_rigid':
            runners.run_ICP_rigid(datasets, task, task_adata, args.method, log_dir, args)
        elif 'ICP' in args.method:
            if args.method == 'ICP_align':
                method_name = f'ICP_align_{args.matching_algo[:5]}_x_{args.xentropy_loss_wt}_reg_{args.l2_reg}'
                method_key = method_name
            else:
                method_name = args.method
            runners.run_ICP_methods(datasets, task, task_adata, method_name, log_dir, args)
        elif args.method == 'ScAlign':
            runners.run_scAlign(datasets, task, task_adata, args.method, log_dir, args)
        elif args.method == 'MNN':
            runners.run_MNN(datasets, task, task_adata, args.method, log_dir, args)
        elif args.method == 'SeuratV3':
            #task_adata = datasets[task.ds_key]
            runners.run_Seurat(datasets, task, task_adata, args.method, log_dir, args)
        task_adata.obsm[method_key+'_TSNE'] = TSNE(n_components=2).fit_transform(task_adata.obsm[method_key])
        task_adata.obsm[method_key+'_PCA'] = PCA(n_components=2, random_state=1373).fit_transform(task_adata.obsm[method_key])
        task_adata.obsm[method_key+'_UMAP'] = umap.UMAP().fit_transform(task_adata.obsm[method_key])
        plot_alignment_results(log_dir, task_adata, method_key, task)
        lisi_score = metrics.lisi2(task_adata.obsm[method_key], task_adata.obs, [task.batch_key, task.ct_key], perplexity=30)
        lisi_score_batch = metrics.lisi2(task_adata.obsm[method_key], task_adata.obs, [task.batch_key], perplexity=30)
        batch_A_adata = task_adata[task_adata.obs[task.batch_key] == task.source_batch, :]
        print(f'batch_A_adata.shape: {batch_A_adata.shape}')
        lisi_score_celltype = metrics.lisi2(batch_A_adata.obsm[method_key], batch_A_adata.obs, [task.ct_key], perplexity=30)
        if args.do_kBET_test:
            try:
                kbet_stats = metrics.kBET(task_adata.obsm[method_key], task_adata.obs, task.batch_key, args.kBET_env_path)
                print(kbet_stats)
                print('kBET medians:')
                print(kbet_stats.median(axis=0))
            except Exception:
                print('kBET failed')
            
    clf_score = None
    if args.method != 'ScAlign' and args.do_clf_score:
        clf_score = metrics.knn_classification_test(task_adata, method_key, task, use_PCA=args.input_space=='PCA')
    result = {
        'lisi': lisi_score,
        'kbet_stats': kbet_stats,
        'clf': clf_score,
        'alignment_task': task,
        'method': args.name,
        'log_dir': log_dir
    }
    if args.method != 'ScAlign' and args.input_space == 'GENE' and args.do_DE_test:
        # Do Differential Expression Test
        de_comparison(task_adata, method_key, task, log_dir)
    print('LISI averages:')
    print('iLISI: {}'.format(lisi_score[task.batch_key].mean()))
    print('cLISI: {}'.format(lisi_score[task.ct_key].mean()))
    print('\nLISI medians:')
    print('iLISI: {}'.format(lisi_score[task.batch_key].median()))
    print('cLISI: {}'.format(lisi_score[task.ct_key].median()))
    print(lisi_score[task.ct_key].quantile([0.25, 0.5, 0.75]))
    print('\n\n')
    print('New LISI')
    print('iLISI: {}'.format(lisi_score_batch[task.batch_key].median()))
    print(lisi_score_batch[task.batch_key].quantile([0.25, 0.5, 0.75]))
    print('cLISI: {}'.format(lisi_score_celltype[task.ct_key].median()))
    print(lisi_score_celltype[task.ct_key].quantile([0.25, 0.5, 0.75]))
    with open(join(log_dir, 'results.pickle'), 'wb') as f:
        pickle.dump(result, f)

    print('DONE')
