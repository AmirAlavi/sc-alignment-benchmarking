from collections import defaultdict

import anndata
import numpy as np
import pandas as pd

adata = anndata.read('all_data_labeled.h5ad')
print(adata.shape)


accessions = []
accessions_cells = []
accessions_n_cell_types = []
accessions_cell_types = []
accessions_cell_type_counts = []
for study in np.unique(adata.obs['accession']):
    study_meta = adata.obs.loc[adata.obs['accession'] == study]

    cells = study_meta.shape[0]
    celltypes, counts = np.unique(study_meta['label_name'], return_counts=True)
    sort_idx = (-counts).argsort()
    counts = counts[sort_idx]
    celltypes = celltypes[sort_idx]
    
    accessions.append(study)
    accessions_cells.append(cells)
    accessions_cell_types.append(celltypes)
    accessions_n_cell_types.append(len(celltypes))
    accessions_cell_type_counts.append(counts)
    
df = pd.DataFrame(data={'study': accessions, 'n_cells': accessions_cells, 'n_cell_types': accessions_n_cell_types, 'cell_types': accessions_cell_types, 'cell_type_counts': accessions_cell_type_counts})

print(df.shape)

print(df.sort_values(by=['n_cells'], ascending=False).head(n=50))

print(df.sort_values(by=['n_cell_types'], ascending=False).head(n=50))


# selected_cell_types = {
#     'CL:0000084': ['GSE89477', 'GSE89405', 'GSE96993'],
#     'CL:0002322': ['GSE81275', 'GSE96986'],
#     'UBERON:0002048': ['GSE69761', 'GSE78045'],
#     'CL:0000037': ['GSE59114', 'GSE68981'],
#     'UBERON:0000966': ['GSE81903', 'GSE80232'],
#     'UBERON:0001264': ['GSE87375', 'GSE78510']
# }

# selected_adata = None
# to_concat = []
# indexer = None
# for ct, studies in selected_cell_types.items():
#     idx = (adata.obs['label_ID'] == ct) & (adata.obs['accession'].isin(studies))
#     if indexer is None:
#         indexer = idx
#     else:
#         indexer = (indexer) | idx
        
#     # ct_adata = adata[(adata.obs['label_ID'] == ct) & (adata.obs['accession'].isin(studies))]
#     # print(ct_adata.obs['label_name'][0])
#     # print(ct_adata.shape)
#     # print(ct_adata.obs.dtypes)
#     # to_concat.append(ct_adata)
#     # # if selected_adata is None:
#     # #     selected_adata = ct_adata
#     # # else:
#     # #     print(selected_adata.obs.dtypes)
#     # #     selected_adata = selected_adata.concatenate(ct_adata)
    
# selected_adata = adata[indexer, :]    
# print(selected_adata.shape)
# selected_adata.write(filename='subset.h5ad', compression='gzip')
