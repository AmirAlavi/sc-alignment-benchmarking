# import pdb; pdb.set_trace()
import sys
#sys.path.append(r'path/to/whiteboard')
import pickle
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


args = {'dataset': 'panc8', 'panc8_n_cell_types': 5, 'filter_hvg': True, 'source': 'indrop1', 'target':'indrop3'}
args = SimpleNamespace(**args)
panc8 = data.get_data(args.dataset, args)
panc8_genes = panc8.var_names
args = {'dataset': 'CellBench', 'panc8_n_cell_types': 5, 'filter_hvg': True, 'source': 'Dropseq', 'target':'10x'}
args = SimpleNamespace(**args)
CellBench = data.get_data(args.dataset, args)
CellBench_genes = CellBench.var_names

model_files = [
    'experiments/CellBench_save_model/ICP-mnn-CellBench-CELseq2-10x/scipr_model.pkl',
    'experiments/CellBench_save_model/ICP-mnn-CellBench-Dropseq-10x/scipr_model.pkl',
    'experiments/panc8_save_model/ICP-mnn-panc8-indrop1-indrop3/scipr_model.pkl',
    'experiments/panc8_save_model/ICP-mnn-panc8-indrop2-indrop3/scipr_model.pkl',
    'experiments/panc8_save_model/ICP-mnn-panc8-indrop4-indrop3/scipr_model.pkl',
]

genes = [CellBench_genes, CellBench_genes, panc8_genes, panc8_genes, panc8_genes]

ref_sets = {
    'GO': go_bp_rs,
    'Reactome': reactome_rs,
    'KEGG': kegg_rs
}

rank_by_smallest_first = False

for model_file, gene_list in zip(model_files, genes):
    print(f'\n\n\n{model_file}')
    with open(model_file, 'rb') as f:
        scipr = pickle.load(f)
        assert(scipr.W_.shape[1] == gene_list.shape[0])
        # W_ is (out_dims, in_genes)
        df = pd.DataFrame(data=scipr.W_.T, index=gene_list, columns=[f'out:{symbol}' for symbol in gene_list])
        df = df.abs()
        ranks = df.rank(axis=0, ascending=rank_by_smallest_first)
        rank_sums = ranks.sum(axis=1)
        rank_sums_sorted = rank_sums.sort_values()
        # print(rank_sums_sorted[:100])
        lim = 100
        threshold = np.mean(rank_sums_sorted[[lim-1, lim]])
        for rs_key, ref_set in ref_sets.items():
            print(rs_key)
            enr = de.enrich.test(ref=ref_set, scores=rank_sums_sorted, gene_ids=rank_sums_sorted.index, clean_ref=True, threshold=threshold)
            if enr.summary().loc[enr.summary()['qval'] < 0.05].shape[0] > 0:
                print(enr.summary().loc[enr.summary()['qval'] < 0.05])
            else:
                print(enr.summary().iloc[:10])
            
            print()