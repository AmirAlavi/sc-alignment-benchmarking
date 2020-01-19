import pandas as pd
import numpy as np
import anndata

import preprocessing

FILTER_MIN_GENES = 1.8e3
FILTER_MIN_READS = 10
FILTER_MIN_DETECTED = 5

def get_data(dataset, args):
    if dataset == 'Kowalcyzk':
        data = get_kowalcyzk()
    elif dataset == 'CellBench':
        data = get_cellbench()
    elif dataset == 'panc8':
        data = get_panc8(args)
    elif dataset == 'panc8-all':
        data = get_panc82()
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
    return data

def get_kowalcyzk():
    # Load and clean
    counts = pd.read_csv('data/Kowalcyzk/Kowalcyzk_counts.csv', index_col=0).T
    meta = pd.read_csv('data/Kowalcyzk/Kowalcyzk_meta.csv', index_col=0)
    counts, meta = preprocessing.clean_counts(counts, meta, FILTER_MIN_GENES, FILTER_MIN_READS, FILTER_MIN_DETECTED)
    adata = anndata.AnnData(X=counts.values, obs=meta)
    print(adata.X.shape)
    ages, ages_counts = np.unique(adata.obs['cell_age'], return_counts=True)
    for age, c in zip(ages, ages_counts):
        print(f'age: {age}, count: {c}') 
    print(adata.obs.info())
    return adata

def get_cellbench():
    protocols = ['10x', 'CELseq2', 'Dropseq']
    adatas = []
    for protocol in protocols:
        print(protocol)
        counts = pd.read_csv('data/CellBench/{}_counts.csv'.format(protocol), index_col=0).T
        counts = counts.loc[:, ~counts.columns.duplicated()]
        meta = pd.read_csv('data/CellBench/{}_meta.csv'.format(protocol), index_col=0)
        counts, meta = preprocessing.remove_doublets(counts, meta)
        counts, meta = preprocessing.clean_counts(counts, meta, FILTER_MIN_GENES, FILTER_MIN_READS, FILTER_MIN_DETECTED)
        adatas.append(anndata.AnnData(X=counts.values, obs=meta, var=pd.DataFrame(index=counts.columns)))
        print(adatas[-1].shape)
        print(np.unique(adatas[-1].obs['cell_line_demuxlet']))
    adata = anndata.AnnData.concatenate(*adatas, join='inner', batch_key='protocol', batch_categories=protocols)
    print(adata.X.shape)
    print(adata.obs.info())
    return adata

def get_panc8(args):
    #protocols = ['celseq', 'celseq2', 'fluidigmc1']
    protocols = ['celseq', 'celseq2', 'smartseq2', 'fluidigmc1', 'indrop1', 'indrop2', 'indrop3', 'indrop4']
    #protocols = ['celseq', 'celseq2', 'smartseq2', 'fluidigmc1']
    # protocols = [args.source, args.target]
    adatas = []
    for protocol in protocols:
        print(protocol)
        counts = pd.read_csv('data/panc8/{}_counts.csv'.format(protocol), index_col=0).T
        counts = counts.loc[:, ~counts.columns.duplicated()]
        meta = pd.read_csv('data/panc8/{}_meta.csv'.format(protocol), index_col=0)
        counts, meta = preprocessing.clean_counts(counts, meta, FILTER_MIN_GENES, FILTER_MIN_READS, FILTER_MIN_DETECTED)
        adatas.append(anndata.AnnData(X=counts.values, obs=meta, var=pd.DataFrame(index=counts.columns)))
        print(adatas[-1].shape)
        print(np.unique(adatas[-1].obs['celltype']))
        #print(adatas[-1].var)
    adata = anndata.AnnData.concatenate(*adatas, join='inner', batch_key='protocol', batch_categories=protocols)
    print(adata.X.shape)
    print(adata.obs.info())
    cell_types, counts = np.unique(adata.obs['celltype'], return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    cell_types = cell_types[sort_idx]
    counts = counts[sort_idx]
    print(cell_types)
    print(counts)
    selector = adata.obs['celltype'].isin(cell_types[:args.panc8_n_cell_types])
    adata = adata[selector]
    return adata

def get_panc82():
    protocols = ['celseq', 'celseq2', 'fluidigmc1']
    #protocols = ['celseq', 'celseq2', 'smartseq2', 'fluidigmc1', 'indrop1', 'indrop2', 'indrop3', 'indrop4']
    adatas = []
    for protocol in protocols:
        print(protocol)
        counts = pd.read_csv('data/panc8/{}_counts.csv'.format(protocol), index_col=0).T
        counts = counts.loc[:, ~counts.columns.duplicated()]
        meta = pd.read_csv('data/panc8/{}_meta.csv'.format(protocol), index_col=0)
        counts, meta = preprocessing.clean_counts(counts, meta, FILTER_MIN_GENES, FILTER_MIN_READS, FILTER_MIN_DETECTED)
        adatas.append(anndata.AnnData(X=counts.values, obs=meta, var=pd.DataFrame(index=counts.columns)))
        print(adatas[-1].shape)
        print(np.unique(adatas[-1].obs['celltype']))
        #print(adatas[-1].var)
    adata = anndata.AnnData.concatenate(*adatas, join='inner', batch_key='protocol', batch_categories=protocols)
    print(adata.X.shape)
    print(adata.obs.info())
    # cell_types, counts = np.unique(adata.obs['celltype'], return_counts=True)
    # sort_idx = np.argsort(counts)[::-1]
    # cell_types = cell_types[sort_idx]
    # counts = counts[sort_idx]
    # print(cell_types)
    # print(counts)
    # selector = adata.obs['celltype'].isin(cell_types[:n_cell_types])
    # adata = adata[selector]
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
