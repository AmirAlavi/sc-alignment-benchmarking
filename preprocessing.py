#import pdb;pdb.set_trace()
import scanpy
# Functions for cleaning the data (filtering)

def remove_doublets(df_counts, df_meta):
    df_counts = df_counts.loc[df_meta['demuxlet_cls'] == 'SNG', :]
    df_meta = df_meta.loc[df_meta['demuxlet_cls'] == 'SNG', :]
    return df_counts, df_meta

''' Remove cells which do not have at least min_genes detected genes
'''
def filter_cells(df_counts, df_meta, min_genes):
    # lib_sizes = df_counts.astype(bool).sum(axis=1)
    # print('Library Sizes of each cell, descriptive statistics:')
    # print(lib_sizes.describe())
    cell_idx = df_counts.astype(bool).sum(axis=1) > min_genes
    df_counts = df_counts.loc[cell_idx, :]
    df_meta = df_meta.loc[cell_idx, :]
    return df_counts, df_meta

''' Remove genes that don't have at least min_reads number of reads
'''
def filter_low_read_genes(df, min_reads):
    df = df.loc[:, df.sum(axis=0) > min_reads]
    return df

''' Remove genes that don't have at least min_cells number of detections
'''
def filter_low_detected_genes(df, min_cells):
    df = df.loc[:, df.astype(bool).sum(axis=0) > min_cells]
    return df


def clean_counts(df_counts, df_meta, min_lib_size, min_reads, min_detected):
    # filter out low-gene cells
    df_counts, df_meta = filter_cells(df_counts, df_meta, min_lib_size)
    # remove genes that don't have many reads
    df_counts = filter_low_read_genes(df_counts, min_reads)
    # remove genes that are not seen in a sufficient number of cells
    df_counts = filter_low_detected_genes(df_counts, min_detected)
    return df_counts, df_meta

def filter_hvg(adata):
    log_normed = scanpy.pp.log1p(adata, copy=True)
    scanpy.pp.highly_variable_genes(log_normed)
    adata.var['highly_variable'] = log_normed.var['highly_variable']
    adata.var['dispersions_norm'] = log_normed.var['dispersions_norm']
    highly_variable = adata.var.index[adata.var['highly_variable'] == True]
    return adata[:, highly_variable]
