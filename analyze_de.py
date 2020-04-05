from types import SimpleNamespace

import numpy as np
import pandas as pd
import diffxpy.api as de

import data

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

GO_BP_GENE_SET_FILE = 'c5.bp.v7.0.symbols.gmt'
REACTOME_GENE_SET_FILE = 'c2.cp.reactome.v7.0.symbols.gmt'
KEGG_GENE_SET_FILE = 'c2.cp.kegg.v7.0.symbols.gmt'

go_bp_rs = de.enrich.RefSets(fn=GO_BP_GENE_SET_FILE)
reactome_rs = de.enrich.RefSets(fn=REACTOME_GENE_SET_FILE)
kegg_rs = de.enrich.RefSets(fn=KEGG_GENE_SET_FILE)

ref_sets = {
    'GO': go_bp_rs,
    'Reactome': reactome_rs,
    'KEGG': kegg_rs
}

def set_group_assignment(adata, cell_type, ct_key):
    group_assignment = [ct if ct == cell_type else 'other' for ct in adata.obs[ct_key]]
    adata.obs['de_group'] = group_assignment

# args = {'dataset': 'panc8', 'panc8_n_cell_types': 5, 'filter_hvg': False, 'source': 'indrop1', 'target':'indrop3'}
# args = SimpleNamespace(**args)

# panc8 = data.get_data(args.dataset, args)

# print('panc8\n\n\n')
# for source in ['indrop1', 'indrop2', 'indrop4']:
#     subset = panc8[(panc8.obs['protocol'] == source) | (panc8.obs['protocol'] == 'indrop3'), :]
#     #subset = panc8[panc8.obs['dataset'] == source, :]
#     print(f'{source}')
#     print(subset.shape)

#     for ct in np.unique(subset.obs['celltype']):
#         print(f'{ct}')
#         set_group_assignment(subset, ct, 'celltype')
#         test_result = de.test.rank_test(subset, grouping='de_group')
#         for rs_key, ref_set in ref_sets.items():
#             print(rs_key)
#             enr = de.enrich.test(ref=ref_set, det=test_result)
#             print(enr.summary().loc[enr.summary()['qval'] < 0.05])
#         print()
#         # B_only_de_results[cell_type] = enr.summary()
#         # enr.summary().to_csv(os.path.join(log_dir, '{}_de_B_only.csv'.format(cell_type)))
#     print()
#     print()
#     print()

args = {'dataset': 'CellBench', 'panc8_n_cell_types': 5, 'filter_hvg': False, 'source': 'Dropseq', 'target':'10x'}
args = SimpleNamespace(**args)

panc8 = data.get_data(args.dataset, args)

print('CellBench\n\n\n')
for source in ['CELseq2', 'Dropseq']:
    subset = panc8[(panc8.obs['protocol'] == source) | (panc8.obs['protocol'] == '10x'), :]
    #subset = panc8[panc8.obs['protocol'] == source, :]
    print(f'{source}')
    print(subset.shape)

    for ct in np.unique(subset.obs['cell_line_demuxlet']):
        print(f'{ct}')
        set_group_assignment(subset, ct, 'cell_line_demuxlet')
        test_result = de.test.rank_test(subset, grouping='de_group')
        for rs_key, ref_set in ref_sets.items():
            print(rs_key)
            enr = de.enrich.test(ref=ref_set, det=test_result)
            print(enr.summary().loc[enr.summary()['qval'] < 0.05])
        print()
        # B_only_de_results[cell_type] = enr.summary()
        # enr.summary().to_csv(os.path.join(log_dir, '{}_de_B_only.csv'.format(cell_type)))
    print()
    print()
    print()

# args = {'dataset': 'pbmcsca_high', 'pbmcsca_high_n_cell_types': 3, 'filter_hvg': False, 'source': '"10x Chromium (v2) A"', 'target': '"10x Chromium (v2)"'}
# args = SimpleNamespace(**args)

# pbmc = data.get_data(args.dataset, args)

# print('pbmc\n\n\n')
# for source in ["10x Chromium (v3)", "10x Chromium (v2) B", "10x Chromium (v2) A"]:
#     subset = pbmc[(pbmc.obs['protocol'] == source) | (pbmc.obs['protocol'] == "10x Chromium (v2)"), :]
#     #subset = pbmc[pbmc.obs['protocol'] == source, :]
#     print(f'{source}')
#     print(subset.shape)

#     for ct in np.unique(subset.obs['CellType']):
#         print(f'{ct}')
#         set_group_assignment(subset, ct, 'CellType')
#         test_result = de.test.rank_test(subset, grouping='de_group')
#         for rs_key, ref_set in ref_sets.items():
#             print(rs_key)
#             enr = de.enrich.test(ref=ref_set, det=test_result)
#             print(enr.summary().loc[enr.summary()['qval'] < 0.05])
#         print()
#         # B_only_de_results[cell_type] = enr.summary()
#         # enr.summary().to_csv(os.path.join(log_dir, '{}_de_B_only.csv'.format(cell_type)))
#     print()
#     print()
#     print()