from types import SimpleNamespace
import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import diffxpy.api as de

import data
from dataset_info import batch_columns, celltype_columns

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

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

sources = {
    'CellBench': ['CELseq2', 'Dropseq'],
    'panc8': ['indrop1', 'indrop2', 'indrop4'],
    'pbmcsca_high': ['10x Chromium (v2) A', '10x Chromium (v2) B', '10x Chromium (v3)']
}

def set_group_assignment(adata, cell_type, ct_key):
    group_assignment = [ct if ct == cell_type else 'other' for ct in adata.obs[ct_key]]
    adata.obs['de_group'] = group_assignment

def clean_up_table_for_printing(table, term_set_name):
    table['set'] = table.apply(lambda row: ' '.join(row['set'].split('_')[1:]), axis=1)
    table.rename(columns={'qval': 'Corrected p-val', 'set': f'{term_set_name} term'}, inplace=True)
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser('analyze-DE', description='Do DE analysis on a dataset')
    parser.add_argument('dataset', help='Which dataset to analyze')
    parser.add_argument('output_folder', help='Path of output folder (created if not exists) to store plots in.')

    # parser.add_argument('--embedding', help='Which type of embedding ot use', choices=['PCA', 'TSNE', 'UMAP'], default='UMAP')
    # parser.add_argument('--rename_method', help='Change the text name of a particular method to appear in the plots.', action='append')
    # parser.add_argument('--rename_dataset', help='Change the text name of a particular dataset to appear in the plots.', action='append')

    args = parser.parse_args()
    de_folder = Path(args.output_folder) / 'differential_expression'
    for path in [args.output_folder, de_folder]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    if args.dataset == 'CellBench':
        source = 'Dropseq'
        target = '10x'
    elif args.dataset == 'panc8':
        source = 'indrop1'
        target = 'indrop3'
    elif args.dataset == 'pbmcsca_high':
        source = '"10x Chromium (v2) A"'
        target = '"10x Chromium (v2)"'
    data_args = {'dataset': args.dataset, 'panc8_n_cell_types': 5, 'pbmcsca_high_n_cell_types': 3, 'filter_hvg': False, 'source': source, 'target': target}
    data_args = SimpleNamespace(**data_args)

    data = data.get_data(data_args.dataset, data_args)
    
    for source in sources[args.dataset]:
        # subset = pbmc[(pbmc.obs['protocol'] == source) | (pbmc.obs['protocol'] == "10x Chromium (v2)"), :]
        subset = data[data.obs[batch_columns[args.dataset]] == source, :]
        print(source)
        print(subset.shape)

        for ct in np.unique(subset.obs[celltype_columns[args.dataset]]):
            print(ct)
            set_group_assignment(subset, ct, celltype_columns[args.dataset])
            test_result = de.test.rank_test(subset, grouping='de_group')
            for rs_key, ref_set in ref_sets.items():
                print(rs_key)
                enr = de.enrich.test(ref=ref_set, det=test_result)
                enr_table = enr.summary().loc[enr.summary()['qval'] < 0.05]
                if enr_table.shape[0] > 0:
                    enr_table = enr_table.head(n=20)[['set', 'qval']]
                    enr_table = clean_up_table_for_printing(enr_table, rs_key)
                    print(enr_table)
                    enr_table.to_latex(de_folder / f'{args.dataset}_{rs_key}_{source}_{ct}.tex', index=False, bold_rows=True)
                else:
                    print('NONE SIGNIFICANT')
            print()
        print()
        print()
        print()
