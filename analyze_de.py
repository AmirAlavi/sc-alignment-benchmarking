from types import SimpleNamespace

import numpy as np
import pandas as pd
import diffxpy.api as de

import data

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

GENE_SET_FILE = 'c5.bp.v7.0.symbols.gmt'
rs = de.enrich.RefSets(fn=GENE_SET_FILE)

def set_group_assignment(adata, cell_type, ct_key):
    group_assignment = [ct if ct == cell_type else 'other' for ct in adata.obs[ct_key]]
    adata.obs['de_group'] = group_assignment

args = {'dataset': 'panc8', 'panc8_n_cell_types': 5, 'filter_hvg': False, 'source': 'indrop1', 'target':'indrop3'}
args = SimpleNamespace(**args)

panc8 = data.get_data(args.dataset, args)

print('panc8\n\n\n')
for source in ['indrop1', 'indrop2', 'indrop4']:
    #subset = panc8[(panc8.obs['protocol'] == 'indrop3') | (panc8.obs['protocol'] == source), :]
    subset = panc8[panc8.obs['dataset'] == source, :]
    print(f'{source}')
    print(subset.shape)

    for ct in np.unique(subset.obs['celltype']):
        print(f'{ct}')
        set_group_assignment(subset, ct, 'celltype')
        test_result = de.test.rank_test(subset, grouping='de_group')
        enr = de.enrich.test(ref=rs, det=test_result)
        print(enr.summary().loc[enr.summary()['qval'] < 0.05])
        print()
        # B_only_de_results[cell_type] = enr.summary()
        # enr.summary().to_csv(os.path.join(log_dir, '{}_de_B_only.csv'.format(cell_type)))
    print()
    print()
    print()

args = {'dataset': 'CellBench', 'panc8_n_cell_types': 5, 'filter_hvg': False, 'source': 'Dropseq', 'target':'10x'}
args = SimpleNamespace(**args)

panc8 = data.get_data(args.dataset, args)

print('CellBench\n\n\n')
for source in ['CELseq2', 'Dropseq']:
    #subset = panc8[(panc8.obs['protocol'] == 'indrop3') | (panc8.obs['protocol'] == source), :]
    subset = panc8[panc8.obs['protocol'] == source, :]
    print(f'{source}')
    print(subset.shape)

    for ct in np.unique(subset.obs['cell_line_demuxlet']):
        print(f'{ct}')
        set_group_assignment(subset, ct, 'cell_line_demuxlet')
        test_result = de.test.rank_test(subset, grouping='de_group')
        enr = de.enrich.test(ref=rs, det=test_result)
        print(enr.summary().loc[enr.summary()['qval'] < 0.05])
        print()
        # B_only_de_results[cell_type] = enr.summary()
        # enr.summary().to_csv(os.path.join(log_dir, '{}_de_B_only.csv'.format(cell_type)))
    print()
    print()
    print()