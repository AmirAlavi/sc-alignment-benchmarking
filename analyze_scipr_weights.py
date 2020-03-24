# import pdb; pdb.set_trace()
import sys
from os.path import join
import glob
from pathlib import Path
#sys.path.append(r'path/to/whiteboard')
import pickle
from types import SimpleNamespace
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import diffxpy.api as de

import data

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# model_files = [
#     'experiments/CellBench_save_model/ICP-mnn-CellBench-CELseq2-10x/scipr_model.pkl',
#     'experiments/CellBench_save_model/ICP-mnn-CellBench-Dropseq-10x/scipr_model.pkl',
#     'experiments/panc8_save_model/ICP-mnn-panc8-indrop1-indrop3/scipr_model.pkl',
#     'experiments/panc8_save_model/ICP-mnn-panc8-indrop2-indrop3/scipr_model.pkl',
#     'experiments/panc8_save_model/ICP-mnn-panc8-indrop4-indrop3/scipr_model.pkl',
# ]


def load_ref_sets():
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
    return ref_sets

def analyze_model_weights_rank_sums(scipr, gene_list, ref_sets, rank_by_smallest_first=False):
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
            print('NONE SIGNIFICANT')
            print(enr.summary().iloc[:10])    
        print()    

def analyze_model_weights_normalized_diag(scipr, gene_list, ref_sets):
    assert(scipr.W_.shape[1] == gene_list.shape[0])
    # W_ is (out_dims, in_genes)
    df = pd.DataFrame(data=scipr.W_.T, index=gene_list, columns=[f'out:{symbol}' for symbol in gene_list])
    df = df.abs()
    colsums = df.sum(axis=0)
    df = df.div(colsums, axis=1)
    diag = np.diag(df)
    sort_idx = (-diag).argsort()[:200]
    weights = diag[sort_idx]
    genes = gene_list[sort_idx]
    for rs_key, ref_set in ref_sets.items():
        print(rs_key)
        enr = de.enrich.test(ref=ref_set, scores=weights, gene_ids=genes, clean_ref=True, all_ids=gene_list)
        if enr.summary().loc[enr.summary()['qval'] < 0.05].shape[0] > 0:
            print(enr.summary().loc[enr.summary()['qval'] < 0.05])
        else:
            print('NONE SIGNIFICANT')
            print(enr.summary().iloc[:10])
            
        print()

def load_models(args):
    #results_by_task = defaultdict(list)
    models = []
    for filename in glob.iglob(join(args.root_folder, '**/scipr_model.pkl'), recursive=True):
        filename = Path(filename)
        print(filename)
        model_items = {}
        model_items['model_file'] = filename
        with open(filename.parent / 'results.pickle', 'rb') as f:
            results = pickle.load(f)
            model_items['results'] = results
            model_items['task'] = results['alignment_task']
        models.append(model_items)
    return models
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('analyze-scipr-weights',
                                     description='Analyze trained SCIPR model weights from experiments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--analysis',
                        help='Type of weight analysis to conduct',
                        choices=['rank_sum', 'diag'],
                        default='rank_sum')
    parser.add_argument('--rank_by_smallest_first',
                        help='In rank_sum analysis, rank by smallest first',
                        action='store_true')
    parser.add_argument('root_folder',
                        help='Root directory to search for model files to analyze.')
    args = parser.parse_args()
    ref_sets = load_ref_sets()
    models = load_models(args)

    cached_gene_lists = {}
    for model_items in models:
        print(f'\n\n\n{model_items["model_file"]}')
        with open(model_items['model_file'], 'rb') as f:
            scipr = pickle.load(f)
        # load data for this model
        task = model_items['task']
        if cached_gene_lists.get(task.ds_key) is not None:
            gene_list = cached_gene_lists.get(task.ds_key)
        else:
            data_args = {'dataset': task.ds_key, 'filter_hvg': True, 'source': task.source_batch, 'target': task.target_batch, 'panc8_n_cell_types': 5, 'pbmcsca_high_n_cell_types': 3}
            data_args = SimpleNamespace(**data_args)
            task_data = data.get_data(data_args.dataset, data_args)
            gene_list = task_data.var_names
            cached_gene_lists[task.ds_key] = gene_list
        
        if args.analysis == 'diag':
            analyze_model_weights_normalized_diag(scipr, gene_list, ref_sets)
        elif args.analysis == 'rank_sum':
            analyze_model_weights_rank_sums(scipr, gene_list, ref_sets, args.rank_by_smallest_first)