#import pdb; pdb.set_trace()
import os
from collections import defaultdict

import numpy as np
import anndata
import diffxpy.api as de
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

GENE_SET_FILE = 'c5.bp.v7.0.symbols.gmt'

def set_group_assignment(adata, cell_type, task_info):
    group_assignment = [ct if ct == cell_type else 'other' for ct in adata.obs[task_info.ct_key]]
    adata.obs['de_group'] = group_assignment


def get_batch_adatas(task_adata, task_info, method_key):
    source_idx = task_adata.obs[task_info.batch_key] == task_info.source_batch
    target_idx = task_adata.obs[task_info.batch_key] == task_info.target_batch
    source_transformed_adata = task_adata.obsm[method_key][source_idx]
    target_adata = task_adata[target_adata]
    return source_transformed_adata, target_adata


def de_comparison(task_adata, method_key, alignment_task, log_dir):
    """Goal: take in an alignment method and two datasets from different domains (aka batches)
    Run DE experiment on expression of one cell type vs others in a single domain
    Then run again with the additional cells from the other domain as well, and get the list of genes for both experiments
    And also do GSEA via GO
    """
    B_only_de_results = {}
    # source_transformed_adata, target_adata = get_batch_adatas(task_adata, alignment_task, method_key)
    adata_idx = task_adata.obs[alignment_task.batch_key] == alignment_task.target_batch
    adata = task_adata[adata_idx]
    rs = de.enrich.RefSets(fn=GENE_SET_FILE)
    print('DEs for target set only...')
    print(adata.shape)
    for cell_type in np.unique(adata.obs[alignment_task.ct_key]):
        print(cell_type)
        set_group_assignment(adata, cell_type, alignment_task)
        test_result = de.test.rank_test(adata, grouping='de_group')
        enr = de.enrich.test(ref=rs, det=test_result, all_ids=adata.var_names)
        print(enr.summary().loc[enr.summary()['qval'] < 0.05])
        B_only_de_results[cell_type] = enr.summary()
        enr.summary().to_csv(os.path.join(log_dir, '{}_de_B_only.csv'.format(cell_type)))

    print('DEs for combined target and source...')
    adata_idx = task_adata.obs[alignment_task.batch_key] == alignment_task.source_batch
    if method_key == 'None_aligned':
        transformed_adata = task_adata[adata_idx]
    else:
        transformed_adata = anndata.AnnData(X=task_adata.obsm[method_key][adata_idx], obs=task_adata.obs[adata_idx], var=task_adata.var)
    # infer labels via k-nn
    print('Real labels')
    print(np.unique(transformed_adata.obs[alignment_task.ct_key], return_counts=True))
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(adata.X, adata.obs[alignment_task.ct_key])
    #pred_target_y = clf.predict(target_x)
    y_true = transformed_adata.obs[alignment_task.ct_key]
    transformed_adata.obs[alignment_task.ct_key] = clf.predict(transformed_adata.X)
    print('Inferred labels')
    print(np.unique(transformed_adata.obs[alignment_task.ct_key], return_counts=True))
    print('1-NN classification accuracy: {}'.format(accuracy_score(y_true, transformed_adata.obs[alignment_task.ct_key])))
    print('Confusion matrix:')
    print('[[TN, FP],\n[FN, TP]]\n')
    print(multilabel_confusion_matrix(y_true, transformed_adata.obs[alignment_task.ct_key]))
    adata = adata.concatenate(transformed_adata)
    print(adata.shape)
    for cell_type in np.unique(adata.obs[alignment_task.ct_key]):
        print(cell_type)
        set_group_assignment(adata, cell_type, alignment_task)
        test_result = de.test.rank_test(adata, grouping='de_group')
        enr = de.enrich.test(ref=rs, det=test_result, all_ids=adata.var_names)
        print(enr.summary().loc[enr.summary()['qval'] < 0.05])
        summary = enr.summary()
        B_only_significant = B_only_de_results[cell_type].loc[B_only_de_results[cell_type]['qval'] < 0.05]
        B_only_set = set(B_only_significant['set'])
        combined_significant = summary.loc[summary['qval'] < 0.05]
        combined_set = set(combined_significant['set'])
        B_only = B_only_set.difference(combined_set)
        combined_only = combined_set.difference(B_only_set)
        shared = B_only_set.intersection(combined_set)
        print('\nB only:')
        print(summary.loc[summary['set'].isin(B_only)])
        print('\nShared:')
        print(summary.loc[summary['set'].isin(shared)])
        print('\nCombined only:')
        print(summary.loc[summary['set'].isin(combined_only)])
        enr.summary().to_csv(os.path.join(log_dir, '{}_de_A_and_B.csv'.format(cell_type)))
            
        
        
