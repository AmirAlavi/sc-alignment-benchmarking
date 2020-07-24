# import pdb; pdb.set_trace()

import pandas as pd
import numpy as np
import anndata

import preprocessing
import dataset_info

FILTER_MIN_GENES = 1.8e3
FILTER_MIN_READS = 10
FILTER_MIN_DETECTED = 5

def get_data_crosstabulation(data, args):
    # Print dataset information for publication:
    meta = data.obs
    celltype_col = dataset_info.celltype_columns[args.dataset]
    batch_col = dataset_info.batch_columns[args.dataset]
    crosstab = pd.crosstab(meta[batch_col], meta[celltype_col])
    crosstab['total'] = crosstab.sum(axis=1)
    crosstab.rename_axis("Cell type", axis="columns", inplace=True)
    crosstab.rename_axis("Batch", inplace=True)
    return crosstab

def print_data_info(data, args):
    print()
    print('-------------------------------')
    print('--------- Dataset Info --------')
    print()
    print(f'\tShape: {data.shape}')
    print()
    print('\tBatches:')
    print()
    batches, counts = np.unique(data.obs[dataset_info.batch_columns[args.dataset]], return_counts=True)
    idx = np.argsort(-counts)
    for b, c in zip(batches[idx], counts[idx]):
        print(f'\t{b}: {c}')
    print()
    print('\tCell Types:')
    print()
    celltypes, counts = np.unique(data.obs[dataset_info.celltype_columns[args.dataset]], return_counts=True)
    idx = np.argsort(-counts)
    for ct, c in zip(celltypes[idx], counts[idx]):
        print(f'\t{ct}: {c}')
    print('-------------------------------')
    print()
    print('----------- Task Info ---------')
    print()
    task_idx = data.obs[dataset_info.batch_columns[args.dataset]].isin([args.source, args.target])
    task_data = data[task_idx, :]
    print(f'\tShape: {task_data.shape}')
    print()
    print('\tCell Types:')
    print()
    print('\t Type \t count in A \t count in B')
    for ct in np.unique(task_data.obs[dataset_info.celltype_columns[args.dataset]]):
        a_count = ((task_data.obs[dataset_info.celltype_columns[args.dataset]] == ct) & (task_data.obs[dataset_info.batch_columns[args.dataset]] == args.source)).sum()
        b_count = ((task_data.obs[dataset_info.celltype_columns[args.dataset]] == ct) & (task_data.obs[dataset_info.batch_columns[args.dataset]] == args.target)).sum()
        print(f'\t{ct}\t{a_count}\t{b_count}')
    print('-------------------------------')
    print()

def get_data(dataset, args):
    if dataset == 'Kowalcyzk':
        data = get_kowalcyzk()
    elif dataset == 'CellBench':
        data = get_cellbench()
    elif dataset == 'panc8':
        data = get_panc8(args)
    elif dataset == 'panc8-all':
        data = get_panc82()
    elif dataset == 'pbmcsca_low':
        data = get_pbmcsca_low(args)
    elif dataset == 'pbmcsca_high':
        data = get_pbmcsca_high(args)
    elif dataset == 'scQuery_retina':
        data = get_scQuery_retina()
    elif dataset == 'scQuery_tcell':
        data = get_scQuery_tcell()
    elif dataset == 'scQuery_lung':
        data = get_scQuery_lung()
    elif dataset == 'scQuery_pancreas':
        data = get_scQuery_pancreas()
    elif dataset == 'scQuery_ESC':
        data = get_scQuery_ESC()
    elif dataset == 'scQuery_HSC':
        data = get_scQuery_HSC()
    elif dataset == 'scQuery_combined':
        data = get_scQuery_combined()
    if args.filter_hvg:
        print('Filter to HVG...')
        print(f'Original       : {data.shape}')
        data = preprocessing.filter_hvg(data)
        print(f'After filtering: {data.shape}')
    elif args.filter_hvg2:
        print('Filter to HVG (2)...')
        print(f'Original       : {data.shape}')
        data = preprocessing.filter_hvg2(data, dataset)
        print(f'After filtering: {data.shape}')
    print_data_info(data, args)
    return data

def get_kowalcyzk():
    # Load and clean
    counts = pd.read_csv('data/Kowalcyzk/Kowalcyzk_counts.csv', index_col=0).T
    meta = pd.read_csv('data/Kowalcyzk/Kowalcyzk_meta.csv', index_col=0)
    counts, meta = preprocessing.clean_counts(counts, meta, FILTER_MIN_GENES, FILTER_MIN_READS, FILTER_MIN_DETECTED)
    adata = anndata.AnnData(X=counts.values, obs=meta)
    # print(adata.X.shape)
    ages, ages_counts = np.unique(adata.obs['cell_age'], return_counts=True)
    # for age, c in zip(ages, ages_counts):
    #     print(f'age: {age}, count: {c}') 
    # print(adata.obs.info())
    return adata

def get_cellbench():
    protocols = ['10x', 'CELseq2', 'Dropseq']
    adatas = []
    for protocol in protocols:
        #print(protocol)
        counts = pd.read_csv('data/CellBench/{}_counts.csv'.format(protocol), index_col=0).T
        counts = counts.loc[:, ~counts.columns.duplicated()]
        meta = pd.read_csv('data/CellBench/{}_meta.csv'.format(protocol), index_col=0)
        counts, meta = preprocessing.remove_doublets(counts, meta)
        counts, meta = preprocessing.clean_counts(counts, meta, FILTER_MIN_GENES, FILTER_MIN_READS, FILTER_MIN_DETECTED)
        adatas.append(anndata.AnnData(X=counts.values, obs=meta, var=pd.DataFrame(index=counts.columns)))
        # print(adatas[-1].shape)
        # print(np.unique(adatas[-1].obs['cell_line_demuxlet']))
    adata = anndata.AnnData.concatenate(*adatas, join='inner', batch_key='protocol', batch_categories=protocols)
    # print(adata.X.shape)
    # print(adata.obs.info())
    return adata

def get_panc8(args):
    #protocols = ['celseq', 'celseq2', 'fluidigmc1']
    protocols = ['celseq', 'celseq2', 'smartseq2', 'fluidigmc1', 'indrop1', 'indrop2', 'indrop3', 'indrop4']
    #protocols = ['celseq', 'celseq2', 'smartseq2', 'fluidigmc1']
    # protocols = [args.source, args.target]
    adatas = []
    for protocol in protocols:
        # print(protocol)
        counts = pd.read_csv('data/panc8/{}_counts.csv'.format(protocol), index_col=0).T
        counts = counts.loc[:, ~counts.columns.duplicated()]
        meta = pd.read_csv('data/panc8/{}_meta.csv'.format(protocol), index_col=0)
        counts, meta = preprocessing.clean_counts(counts, meta, FILTER_MIN_GENES, FILTER_MIN_READS, FILTER_MIN_DETECTED)
        adatas.append(anndata.AnnData(X=counts.values, obs=meta, var=pd.DataFrame(index=counts.columns)))
        # print(adatas[-1].shape)
        # print(np.unique(adatas[-1].obs['celltype']))
        #print(adatas[-1].var)
    adata = anndata.AnnData.concatenate(*adatas, join='inner', batch_key='protocol', batch_categories=protocols)
    # print(adata.X.shape)
    # print(adata.obs.info())
    cell_types, counts = np.unique(adata.obs['celltype'], return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    cell_types = cell_types[sort_idx]
    counts = counts[sort_idx]
    # print(cell_types)
    # print(counts)
    selector = adata.obs['celltype'].isin(cell_types[:args.panc8_n_cell_types])
    adata = adata[selector]
    return adata

def get_panc82():
    protocols = ['celseq', 'celseq2', 'fluidigmc1']
    #protocols = ['celseq', 'celseq2', 'smartseq2', 'fluidigmc1', 'indrop1', 'indrop2', 'indrop3', 'indrop4']
    adatas = []
    for protocol in protocols:
        # print(protocol)
        counts = pd.read_csv('data/panc8/{}_counts.csv'.format(protocol), index_col=0).T
        counts = counts.loc[:, ~counts.columns.duplicated()]
        meta = pd.read_csv('data/panc8/{}_meta.csv'.format(protocol), index_col=0)
        counts, meta = preprocessing.clean_counts(counts, meta, FILTER_MIN_GENES, FILTER_MIN_READS, FILTER_MIN_DETECTED)
        adatas.append(anndata.AnnData(X=counts.values, obs=meta, var=pd.DataFrame(index=counts.columns)))
        # print(adatas[-1].shape)
        # print(np.unique(adatas[-1].obs['celltype']))
        #print(adatas[-1].var)
    adata = anndata.AnnData.concatenate(*adatas, join='inner', batch_key='protocol', batch_categories=protocols)
    # print(adata.X.shape)
    # print(adata.obs.info())
    # cell_types, counts = np.unique(adata.obs['celltype'], return_counts=True)
    # sort_idx = np.argsort(counts)[::-1]
    # cell_types = cell_types[sort_idx]
    # counts = counts[sort_idx]
    # print(cell_types)
    # print(counts)
    # selector = adata.obs['celltype'].isin(cell_types[:n_cell_types])
    # adata = adata[selector]
    return adata

def get_pbmcsca_low(args):
    protocols = ['Smart-seq2', 'CEL-Seq2']
    adatas = []
    for protocol in protocols:
        # print(protocol)
        counts = pd.read_csv('data/pbmcsca/{}_counts.csv'.format(protocol), index_col=0).T
        counts = counts.loc[:, ~counts.columns.duplicated()]
        meta = pd.read_csv('data/pbmcsca/{}_meta.csv'.format(protocol), index_col=0)
        counts, meta = preprocessing.clean_counts(counts, meta, FILTER_MIN_GENES, FILTER_MIN_READS, FILTER_MIN_DETECTED)
        adatas.append(anndata.AnnData(X=counts.values, obs=meta, var=pd.DataFrame(index=counts.columns)))
        # print(adatas[-1].shape)
        # print(np.unique(adatas[-1].obs['celltype']))
        #print(adatas[-1].var)
    adata = anndata.AnnData.concatenate(*adatas, join='inner', batch_key='protocol', batch_categories=protocols)
    # print(adata.X.shape)
    # print(adata.obs.info())
    cell_types, counts = np.unique(adata.obs['CellType'], return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    cell_types = cell_types[sort_idx]
    counts = counts[sort_idx]
    # print(cell_types)
    # print(counts)
    selector = adata.obs['CellType'].isin(cell_types[:args.pbmcsca_high_n_cell_types])
    adata = adata[selector]
    return adata

def get_pbmcsca_high(args):
    protocols = ["10x Chromium (v2) A", "10x Chromium (v2) B", "10x Chromium (v3)", "Drop-seq", "Seq-Well", "inDrops", "10x Chromium (v2)"]
    adatas = []
    for protocol in protocols:
        # print(protocol)
        counts = pd.read_csv('data/pbmcsca/{}_counts.csv'.format(protocol), index_col=0).T
        counts = counts.loc[:, ~counts.columns.duplicated()]
        meta = pd.read_csv('data/pbmcsca/{}_meta.csv'.format(protocol), index_col=0)
        counts, meta = preprocessing.clean_counts(counts, meta, 250, FILTER_MIN_READS, FILTER_MIN_DETECTED)
        adatas.append(anndata.AnnData(X=counts.values, obs=meta, var=pd.DataFrame(index=counts.columns)))
        # print(adatas[-1].shape)
        # print(np.unique(adatas[-1].obs['celltype']))
        #print(adatas[-1].var)
    adata = anndata.AnnData.concatenate(*adatas, join='inner', batch_key='protocol', batch_categories=protocols)
    # print(adata.X.shape)
    # print(adata.obs.info())
    cell_types, counts = np.unique(adata.obs['CellType'], return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    cell_types = cell_types[sort_idx]
    counts = counts[sort_idx]
    # print(cell_types)
    # print(counts)
    selector = adata.obs['CellType'].isin(cell_types[:args.pbmcsca_high_n_cell_types])
    adata = adata[selector]
    return adata

def get_scQuery_retina():
    adata = anndata.read('data/scQuery/subset_int.h5ad')
    adata = adata[adata.obs['label_ID'] == 'UBERON:0000966', :]
    return adata

def get_scQuery_tcell():
    adata = anndata.read('data/scQuery/subset_int.h5ad')
    adata = adata[adata.obs['label_ID'] == 'CL:0000084', :]
    return adata

def get_scQuery_lung():
    adata = anndata.read('data/scQuery/subset_int.h5ad')
    adata = adata[adata.obs['label_ID'] == 'UBERON:0002048', :]
    return adata

def get_scQuery_pancreas():
    adata = anndata.read('data/scQuery/subset_int.h5ad')
    adata = adata[adata.obs['label_ID'] == 'UBERON:0001264', :]
    return adata

def get_scQuery_ESC():
    adata = anndata.read('data/scQuery/subset_int.h5ad')
    adata = adata[adata.obs['label_ID'] == 'CL:0002322', :]
    return adata

def get_scQuery_HSC():
    adata = anndata.read('data/scQuery/subset_int.h5ad')
    adata = adata[adata.obs['label_ID'] == 'CL:0000037', :]
    return adata

def get_scQuery_combined():
    adata = anndata.read('data/scQuery/artificial_HSC_ESC_pancreas.h5ad')
    return adata
