from types import SimpleNamespace

import numpy as np
import pandas as pd
import diffxpy.api as de

import data
import dataset_info
import classification_test

# GENE_SET_FILE = 'c5.bp.v7.0.symbols.gmt'
# rs = de.enrich.RefSets(fn=GENE_SET_FILE)

def set_group_assignment(adata, cell_type, ct_key):
    group_assignment = [ct if ct == cell_type else 'other' for ct in adata.obs[ct_key]]
    adata.obs['de_group'] = group_assignment



scipr_model_paths = {
    'panc8': {
        'indrop1': 'experiments/ICP-mnn-panc8-indrop1-indrop3/scipr_model.pkl',
        'indrop2': 'experiments/ICP-mnn-panc8-indrop2-indrop3/scipr_model.pkl',
        'indrop4': 'experiments/ICP-mnn-panc8-indrop4-indrop3/scipr_model.pkl'
    }
    'CellBench': {
        'Dropseq': 'experiments/ICP-mnn-CellBench-Dropseq-10x/scipr_model.pkl',
        'CELseq2': 'experiments/ICP-mnn-CellBench-CELseq2-10x/scipr_model.pkl'
    }
}

source_batches = {
    'panc8': ['indrop1', 'indrop2', 'indrop4'],
    'CellBench': ['Dropseq', 'CELseq2']
}

classification_args ={
    'panc8': {

    }
    ''
}

clf_args = {
    'ref_batch': 'experiments/CellBench/ICP-mnn-CellBench-Dropseq-10x/target_x.pkl',
    'ref_batch_y': 'experiments/CellBench/ICP-mnn-CellBench-Dropseq-10x/target_y.pkl',
    'batch2_unaligned': 'experiments/CellBench/ICP-mnn-CellBench-CELseq2-10x/source_unaligned_x.pkl',
    'batch2_aligned': 'experiments/CellBench/ICP-mnn-CellBench-CELseq2-10x/source_aligned_x.pkl',
    'batch2_y': 'experiments/CellBench/ICP-mnn-CellBench-CELseq2-10x/source_y.pkl',
    'test_batch_unaligned': 'experiments/CellBench/ICP-mnn-CellBench-Dropseq-10x/source_unaligned_x.pkl',
    'test_batch_aligned': 'experiments/CellBench/ICP-mnn-CellBench-Dropseq-10x/source_aligned_x.pkl',
    'test_batch_y': 'experiments/CellBench/ICP-mnn-CellBench-Dropseq-10x/source_y.pkl'
}

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

def load_all_data():
    args = {'dataset': 'panc8', 'panc8_n_cell_types': 5, 'filter_hvg': False, 'source': 'indrop1', 'target':'indrop3'}
    args = SimpleNamespace(**args)
    panc8 = data.get_data(args.dataset, args)
    args = {'dataset': 'CellBench', 'panc8_n_cell_types': 5, 'filter_hvg': False, 'source': 'Dropseq', 'target':'10x'}
    args = SimpleNamespace(**args)
    CellBench = data.get_data(args.dataset, args)
    data = {
        'panc8': panc8,
        'CellBench': Cellbench
    }
    return data

datasets = load_all_data()

for ds_key, dataset in datasets.items():
    for source in source_batches[dataset]:
        
        subset = dataset[dataset.obs[dataset_info.batch_columns[ds_key]] == source, :]
        print(f'{source}')
        print(subset.shape)


        # Do DE for cell types
        for ct in np.unique(subset.obs[dataset_info.celltype_columns[ds_key]]):
            print(f'{ct}')
            set_group_assignment(subset, ct, dataset_info.celltype_columns[ds_key])
            test_result = de.test.rank_test(subset, grouping='de_group')
            
            for rs_key, ref_set in ref_sets.items():
                enr = de.enrich.test(ref=ref_set, det=test_result)
                print(enr.summary().loc[enr.summary()['qval'] < 0.05])
                print()
                # B_only_de_results[cell_type] = enr.summary()
                # enr.summary().to_csv(os.path.join(log_dir, '{}_de_B_only.csv'.format(cell_type)))
        
        # Find top genes in scipr_model
        model_path = scipr_model_paths[ds_key][source]
        scipr = pickle.load(model_path)
        assert(scipr.W_.shape[1] == dataset.var_names.shape[0])
        # W_ is (out_dims, in_genes)
        df = pd.DataFrame(data=scipr.W_.T, index=dataset.var_names, columns=[f'out:{symbol}' for symbol in dataset.var_names])
        df = df.abs()
        ranks = df.rank(axis=0, ascending=False)
        rank_sums = ranks.sum(axis=1)
        rank_sums_sorted = rank_sums.sort_values()
        # print(rank_sums_sorted[:100])
        threshold = np.mean(rank_sums_sorted[[100, 101]])
        for rs_key, ref_set in ref_sets.items():
            enr = de.enrich.test(ref=ref_set, scores=rank_sums_sorted, gene_ids=rank_sums_sorted.index, threshold=threshold)
            print(enr.summary().loc[enr.summary()['qval'] < 0.05])
        
        # Conduct classification test
        args = {
            'ref_batch': '',
            'ref_batch_y': '',
            'batch2_unaligned': '',
            'batch2_aligned': '',
            'batch2_y': '',
            'test_batch_unaligned': '',
            'test_batch_aligned': '',
            'test_batch_y': ''
        }
        args = SimpleNamespace(**args)
        classification_test.test_no_preprocessing(args)
        print()
        print()
        print()