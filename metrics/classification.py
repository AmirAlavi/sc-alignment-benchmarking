from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import KNeighborsClassifier
from geosketch import gs
from fbpca import pca
import pandas as pd
import numpy as np


def get_source_target(task_adata, method_key, task_info, use_PCA=False, test_adata=None):
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

    if test_adata is not None:
        if use_PCA:
            test_X = test_adata.obsm['PCA']
        else:
            test_X = test_adata.X
        test_y = test_adata.obs[task_info.ct_key]
        test_y = label_encoder.transform(test_y)
        return source_aligned_x, source_x, target_x, source_y, target_y, test_X, test_y

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

def knn_classification_test(task_adata, method_key, alignment_task, use_PCA=False):
    #import pdb; pdb.set_trace()
    # Train a classifier on just the Target batch
    # Then compare performance of applying it on raw source batch
    # versus on transformed source batch
    print('Computing classification performance...')
    source_aligned_x, source_x, target_x, source_y, target_y = get_source_target(task_adata, method_key, alignment_task, use_PCA)

    #clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    clf = KNeighborsClassifier()
    clf.fit(target_x, target_y)
    #pred_target_y = clf.predict(target_x)
    pred_source_y = clf.predict(source_x)
    pred_source_aligned_y = clf.predict(source_aligned_x)

    #target_acc = accuracy_score(target_y, pred_target_y)
    source_acc = accuracy_score(source_y, pred_source_y)
    source_aligned_acc = accuracy_score(source_y, pred_source_aligned_y)

    # score_target_y = clf.predict_proba(target_x)
    # score_source_y = clf.predict_proba(source_x)
    # score_source_aligned_y = clf.predict_proba(source_aligned_x)

    # target_auc = roc_auc_score(target_y, score_target_y)
    # source_auc = roc_auc_score(source_y, score_source_y)
    # source_aligned_auc = roc_auc_score(source_y, score_source_aligned_y)

    scores = {
#        'target_acc': target_acc,
        'source_acc': source_acc,
        'source_aligned_acc': source_aligned_acc,
        # 'target_auc': target_auc,
        # 'source_auc': source_auc,
        # 'source_aligned_auc': source_aligned_auc
    }
    return scores

def geosketch_sample_dimred(X, n):
    U, s, Vt = pca(X, k=100) # E.g., 100 PCs.
    X_dimred = U[:, :100] * s[:100]
    sketch_index = gs(X_dimred, n, replace=False)
    return X_dimred[sketch_index]

def paired_batch_classification_test(test_batch, task_adata, method_key, alignment_task, use_PCA=False):
    source_aligned_x, source_x, target_x, source_y, target_y, test_X, test_y = get_source_target(task_adata, method_key, alignment_task, use_PCA, test_batch)
    source_x = normalize(source_x)
    target_x = normalize(target_x)
    test_X = normalize(test_X)
    # clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    train_size = min(source_x.shape[0], target_x.shape[0])
    print('### PAIRED BATCH CLASSIFICATION TESTS')
    print(f'train size: {train_size}\n')
    def train_clf_helper(X_tr, y_tr, X_te, y_te):
        if train_size < X_tr.shape[0]:
            sketch_idx = gs(X_tr, train_size, replace=False)
            X_tr = X_tr[sketch_idx]
            y_tr = y_tr[sketch_idx]
        # clf = KNeighborsClassifier()
        clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        print(classification_report(y_te, y_pred))
        print()
        return classification_report(y_te, y_pred, output_dict=True)

    reports = {
        'model(A)': train_clf_helper(source_x, source_y, test_X, test_y),
        'model(B)': train_clf_helper(target_x, target_y, test_X, test_y),
        'model(A+B)': train_clf_helper(np.concatenate([source_x, target_x], axis=0), np.concatenate([source_y, target_y]), test_X, test_y),
        'model(f(A)+B)': train_clf_helper(np.concatenate([source_aligned_x, target_x], axis=0), np.concatenate([source_y, target_y]), test_X, test_y)
    }
    return reports

