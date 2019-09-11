import pandas as pd
import numpy as np
import anndata

import preprocessing

FILTER_MIN_GENES = 1.8e3
FILTER_MIN_READS = 10
FILTER_MIN_DETECTED = 5

def get_data(dataset):
    if dataset == 'Kowalcyzk':
        return get_kowalcyzk()
    elif dataset == 'CellBench':
        return get_cellbench()
    elif dataset == 'panc8':
        return get_panc8()

def get_kowalcyzk():
    # Load and clean
    counts = pd.read_csv('data/Kowalcyzk/Kowalcyzk_counts.csv', index_col=0).T
    meta = pd.read_csv('data/Kowalcyzk/Kowalcyzk_meta.csv', index_col=0)
    counts, meta = preprocessing.clean_counts(counts, meta, FILTER_MIN_GENES, FILTER_MIN_READS, FILTER_MIN_DETECTED)
    adata = anndata.AnnData(X=counts.values, obs=meta)
    print(adata.X.shape)
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

def get_panc8(n_cell_types=5):
    protocols = ['celseq', 'celseq2', 'fluidigmc1']
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
    selector = adata.obs['celltype'].isin(cell_types[:n_cell_types])
    adata = adata[selector]
    return adata