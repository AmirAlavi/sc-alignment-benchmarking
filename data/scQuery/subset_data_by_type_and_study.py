from collections import defaultdict

import anndata
import numpy as np
import pandas as pd

adata = anndata.read('all_data_labeled.h5ad')
print(adata.shape)

celltypes = []
celltype_names = []
n_cells = []
n_studies = []
study_sizes = []
studies = []

for celltype in np.unique(adata.obs['label_ID']):
    ctype_meta = adata.obs.loc[adata.obs['label_ID'] == celltype]
    ctype_name = ctype_meta['label_name'].iloc[0]
    ctype_cells = ctype_meta.shape[0]
    ctype_studies, ctype_studies_sizes = np.unique(ctype_meta['accession'], return_counts=True)
    sort_idx = (-ctype_studies_sizes).argsort()
    ctype_studies_sizes = ctype_studies_sizes[sort_idx]
    ctype_studies = ctype_studies[sort_idx]

    
    
    celltypes.append(celltype)
    celltype_names.append(ctype_name)
    n_cells.append(ctype_cells)
    n_studies.append(len(ctype_studies))
    study_sizes.append(ctype_studies_sizes)
    studies.append(ctype_studies)

df = pd.DataFrame(data={'celltype': celltypes, 'name': celltype_names, 'n_cells': n_cells, 'n_studies': n_studies, 'studies': studies, 'study_sizes': study_sizes})

print(df.shape)

print(df.sort_values(by=['n_cells'], ascending=False).head(n=50))

print(df.sort_values(by=['n_studies'], ascending=False).head(n=50))


selected_cell_types = {
    'CL:0000084': ['GSE89477', 'GSE89405', 'GSE96993'],
    'CL:0002322': ['GSE81275', 'GSE96986'],
    'UBERON:0002048': ['GSE69761', 'GSE78045'],
    'CL:0000037': ['GSE59114', 'GSE68981'],
    'UBERON:0000966': ['GSE81903', 'GSE80232'],
    'UBERON:0001264': ['GSE87375', 'GSE78510']
}

selected_adata = None
to_concat = []
indexer = None
for ct, studies in selected_cell_types.items():
    idx = (adata.obs['label_ID'] == ct) & (adata.obs['accession'].isin(studies))
    if indexer is None:
        indexer = idx
    else:
        indexer = (indexer) | idx
        
    # ct_adata = adata[(adata.obs['label_ID'] == ct) & (adata.obs['accession'].isin(studies))]
    # print(ct_adata.obs['label_name'][0])
    # print(ct_adata.shape)
    # print(ct_adata.obs.dtypes)
    # to_concat.append(ct_adata)
    # # if selected_adata is None:
    # #     selected_adata = ct_adata
    # # else:
    # #     print(selected_adata.obs.dtypes)
    # #     selected_adata = selected_adata.concatenate(ct_adata)
    
selected_adata = adata[indexer, :]    
print(selected_adata.shape)
selected_adata.write(filename='subset.h5ad', compression='gzip')
