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
print(type(panc8))

print(panc8.var_names)

model_files = [
    'experiments/panc8_save_model/ICP-mnn-panc8-indrop1-indrop3/scipr_model.pkl',
    'experiments/panc8_save_model/ICP-mnn-panc8-indrop2-indrop3/scipr_model.pkl',
    'experiments/panc8_save_model/ICP-mnn-panc8-indrop4-indrop3/scipr_model.pkl'
]

for model_file in model_files:
    print(f'\n\n\n{model_file}')
    with open(model_file, 'rb') as f:
        scipr = pickle.load(f)
        assert(scipr.W_.shape[1] == panc8.var_names.shape[0])
        # W_ is (out_dims, in_genes)
        df = pd.DataFrame(data=scipr.W_.T, index=panc8.var_names, columns=[f'out:{symbol}' for symbol in panc8.var_names])
        df = df.abs()
        ranks = df.rank(axis=0, ascending=False)
        rank_sums = ranks.sum(axis=1)
        rank_sums_sorted = rank_sums.sort_values()
        # print(rank_sums_sorted[:100])
        threshold = np.mean(rank_sums_sorted[[100, 101]])
        enr = de.enrich.test(ref=go_bp_rs, scores=rank_sums_sorted, gene_ids=rank_sums_sorted.index, threshold=threshold)
        print(enr.summary().loc[enr.summary()['qval'] < 0.05])