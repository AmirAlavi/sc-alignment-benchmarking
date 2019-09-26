from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def get_source_target(task_adata, method_key, task_info, use_PCA=False):
    """Get the source data to be projected, as well as the target data on which it will be projected,
    both as np.ndarrays.
    """
    source_idx = task_adata.obs[task_info.batch_key] == task_info.source_batch
    if task_info.leave_out_ct is not None:
        target_idx = (task_adata.obs[task_info.batch_key] == task_info.target_batch) & (task_adata.obs[task_info.ct_key] != task_info.leave_out_ct)
    else:
        target_idx = task_adata.obs[task_info.batch_key] == task_info.target_batch
    if use_PCA:
        source_x = task_adata.obsm['PCA'][source_idx]
        target_x = task_adata.obsm['PCA'][target_idx]
    else:
        source_x = task_adata.X[source_idx]
        target_x = task_adata.X[target_idx]
    if 'none' in method_key.lower():
        if use_PCA:
            source_aligned_x = task_adata.obsm['PCA'][source_idx]
        else:
            source_aligned_x = task_adata.X[source_idx]
    else:
        source_aligned_x = task_adata.obsm[method_key][source_idx]
    source_y = task_adata.obs[task_info.ct_key][source_idx]
    target_y = task_adata.obs[task_info.ct_key][target_idx]
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate((source_y, target_y), axis=None))
    source_y = label_encoder.transform(source_y)
    target_y = label_encoder.transform(target_y)
    return source_aligned_x, source_x, target_x, source_y, target_y

def classification_test(task_adata, method_key, alignment_task, use_PCA=False):
    #import pdb; pdb.set_trace()
    # Train a classifier on just the Target batch
    # Then compare performance of applying it on raw source batch
    # versus on transformed source batch
    print('Computing classification performance...')
    source_aligned_x, source_x, target_x, source_y, target_y = get_source_target(task_adata, method_key, alignment_task, use_PCA)

    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    clf.fit(target_x, target_y)
    pred_target_y = clf.predict(target_x)
    pred_source_y = clf.predict(source_x)
    pred_source_aligned_y = clf.predict(source_aligned_x)

    target_acc = accuracy_score(target_y, pred_target_y)
    source_acc = accuracy_score(source_y, pred_source_y)
    source_aligned_acc = accuracy_score(source_y, pred_source_aligned_y)

    # score_target_y = clf.predict_proba(target_x)
    # score_source_y = clf.predict_proba(source_x)
    # score_source_aligned_y = clf.predict_proba(source_aligned_x)

    # target_auc = roc_auc_score(target_y, score_target_y)
    # source_auc = roc_auc_score(source_y, score_source_y)
    # source_aligned_auc = roc_auc_score(source_y, score_source_aligned_y)

    scores = {
        'target_acc': target_acc,
        'source_acc': source_acc,
        'source_aligned_acc': source_aligned_acc,
        # 'target_auc': target_auc,
        # 'source_auc': source_auc,
        # 'source_aligned_auc': source_aligned_auc
    }
    return scores

