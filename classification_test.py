import argparse
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from geosketch import gs

# def paired_batch_classification_test(ref_x, ref_x2, ref_y, batch2_x, batch2_y, f_batch2_x, test_x, test_y, f_test_x):
#     # clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
#     train_size = min(ref_x.shape[0], batch2_x.shape[0])
#     print('### PAIRED BATCH CLASSIFICATION TESTS')
#     print(f'train size: {train_size}\n')
#     def train_clf_helper(X_tr, y_tr, X_te, y_te):
#         if train_size < X_tr.shape[0]:
#             sketch_idx = gs(X_tr, train_size, replace=False)
#             X_tr = X_tr[sketch_idx]
#             y_tr = y_tr[sketch_idx]
#         # clf = KNeighborsClassifier()
#         clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
#         clf.fit(X_tr, y_tr)
#         y_pred = clf.predict(X_te)
#         print(classification_report(y_te, y_pred))
#         print()
#         return classification_report(y_te, y_pred, output_dict=True)

#     # reports = {
#     #     'model(A),Y': train_clf_helper(ref_x, ref_y, test_x, test_y),
#     #     'model(B),Y': train_clf_helper(batch2_x, batch2_y, test_x, test_y),
#     #     'model(A+B),Y': train_clf_helper(np.concatenate([ref_x, batch2_x], axis=0), np.concatenate([ref_y, batch2_y]), test_x, test_y),
#     #     'model(f(A)+B),Y': train_clf_helper(np.concatenate([ref_x, f_batch2_x], axis=0), np.concatenate([ref_y, batch2_y]), test_x, test_y),
#     # }
#     print()
#     print('With aligned test batch:')
#     print()
#     # reports['model(A),f(Y)'] = train_clf_helper(ref_x, ref_y, f_test_x, test_y)
#     # reports['model(B),f(Y)'] = train_clf_helper(batch2_x, batch2_y, f_test_x, test_y)
#     # reports['model(A+B),f(Y)'] = train_clf_helper(np.concatenate([ref_x, batch2_x], axis=0), np.concatenate([ref_y, batch2_y]), f_test_x, test_y)
#     # reports['model(f(A)+B),f(Y)'] = train_clf_helper(np.concatenate([ref_x, f_batch2_x], axis=0), np.concatenate([ref_y, batch2_y]), f_test_x, test_y)
#     # print()
#     # print('With aligned test batch (2):')
#     # print()
#     reports = {}
#     reports['model(A),Y'] = train_clf_helper(ref_x, ref_y, test_x, test_y)
#     reports['model(B),Y'] = train_clf_helper(batch2_x, batch2_y, test_x, test_y)
#     reports['model(A+B),Y'] = train_clf_helper(np.concatenate([ref_x, batch2_x], axis=0), np.concatenate([ref_y, batch2_y]), test_x, test_y)
#     reports['model(f(A)+B),f(Y)'] = train_clf_helper(np.concatenate([ref_x2, f_batch2_x], axis=0), np.concatenate([ref_y, batch2_y]), f_test_x, test_y)
#     return reports

def paired_batch_classification_test_no_prep(ref_x, ref_x2, ref_y, batch2_x, batch2_y, f_batch2_x, test_x, test_y, f_test_x, args):
    # clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    train_size = min(ref_x.shape[0], batch2_x.shape[0])
    print('### PAIRED BATCH CLASSIFICATION TESTS')
    print(f'train size: {train_size}\n')
    def train_clf_helper(X_tr, y_tr, X_te, y_te):
        if train_size < X_tr.shape[0]:
            sketch_idx = gs(X_tr, train_size, replace=False)
            X_tr = X_tr[sketch_idx]
            y_tr = y_tr[sketch_idx]
        # clf = KNeighborsClassifier()
        clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
        print(f'fit on {X_tr.shape[0]} samples, test on {y_tr.shape[0]}')
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        print(classification_report(y_te, y_pred))
        print()
        return classification_report(y_te, y_pred, output_dict=True)

    print()
    print('With aligned test batch:')
    print()
    reports = {}
    reports['model(A),Y'] = train_clf_helper(ref_x, ref_y, test_x, test_y)
    reports['model(B),Y'] = train_clf_helper(batch2_x, batch2_y, test_x, test_y)
    reports['model(A+B),Y'] = train_clf_helper(np.concatenate([ref_x, batch2_x], axis=0), np.concatenate([ref_y, batch2_y]), test_x, test_y)
    reports['model(f(A)+B),f(Y)'] = train_clf_helper(np.concatenate([apply_normalization(ref_x2, args), f_batch2_x], axis=0),
                                                     np.concatenate([ref_y, batch2_y]), f_test_x, test_y)
    return reports

def apply_normalization(X, args):
    if args.input_normalization == 'std':
        print('Applying Standard Scaling')
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        return X, scaler
    elif args.input_normalization == 'l2':
        print('Applying L2 Normalization')
        return normalize(X)
    elif args.input_normalization == 'log':
        print('Applying log normalization')
        return np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)

# def test_with_pca(args):
#     with open(args.ref_batch, 'rb') as f:
#         ref_x = pickle.load(f)
#     with open(args.ref_batch_y, 'rb') as f:
#         ref_y = pickle.load(f)
#     with open(args.batch2_unaligned, 'rb') as f:
#         batch2_x = pickle.load(f)
#     with open(args.batch2_y, 'rb') as f:
#         batch2_y = pickle.load(f)
#     with open(args.batch2_aligned, 'rb') as f:
#         f_batch2_x = pickle.load(f)
#     with open(args.test_batch_unaligned, 'rb') as f:
#         test_x = pickle.load(f)
#     with open(args.test_batch_y, 'rb') as f:
#         test_y = pickle.load(f)
#     with open(args.test_batch_aligned, 'rb') as f:
#         f_test_x = pickle.load(f)

#     ref_x = apply_normalization(ref_x, args)
#     batch2_x = apply_normalization(batch2_x, args)
#     test_x = apply_normalization(test_x, args)

#     pca = PCA(n_components=100, random_state=1373).fit(np.concatenate([ref_x, batch2_x, test_x], axis=0))
#     ref_x_pca = pca.transform(ref_x)
#     batch2_x_pca = pca.transform(batch2_x)
#     test_x_pca = pca.transform(test_x)

#     pca = PCA(n_components=100, random_state=1373).fit(np.concatenate([ref_x, f_batch2_x, f_test_x], axis=0))
#     ref_x_pca2 = pca.transform(ref_x)
#     f_batch2_x_pca = pca.transform(f_batch2_x)
#     f_test_x_pca = pca.transform(f_test_x)
#     paired_batch_classification_test(ref_x_pca,
#                                      ref_x_pca2,
#                                      ref_y,
#                                      batch2_x_pca,
#                                      batch2_y,
#                                      f_batch2_x_pca,
#                                      test_x_pca,
#                                      test_y,
#                                      f_test_x_pca)

# def test_no_pca(args):
#     with open(args.ref_batch, 'rb') as f:
#         ref_x = pickle.load(f)
#     with open(args.ref_batch_y, 'rb') as f:
#         ref_y = pickle.load(f)
#     with open(args.batch2_unaligned, 'rb') as f:
#         batch2_x = pickle.load(f)
#     with open(args.batch2_y, 'rb') as f:
#         batch2_y = pickle.load(f)
#     with open(args.batch2_aligned, 'rb') as f:
#         f_batch2_x = pickle.load(f)
#     with open(args.test_batch_unaligned, 'rb') as f:
#         test_x = pickle.load(f)
#     with open(args.test_batch_y, 'rb') as f:
#         test_y = pickle.load(f)
#     with open(args.test_batch_aligned, 'rb') as f:
#         f_test_x = pickle.load(f)

#     ref_x = apply_normalization(ref_x, args)
#     batch2_x = apply_normalization(batch2_x, args)
#     test_x = apply_normalization(test_x, args)

#     paired_batch_classification_test(ref_x,
#                                      ref_x,
#                                      ref_y,
#                                      batch2_x,
#                                      batch2_y,
#                                      f_batch2_x,
#                                      test_x,
#                                      test_y,
#                                      f_test_x)

def test_no_preprocessing(args):
    with open(args.ref_batch, 'rb') as f:
        ref_x = pickle.load(f)
    with open(args.ref_batch_y, 'rb') as f:
        ref_y = pickle.load(f)
    with open(args.batch2_unaligned, 'rb') as f:
        batch2_x = pickle.load(f)
    with open(args.batch2_y, 'rb') as f:
        batch2_y = pickle.load(f)
    with open(args.batch2_aligned, 'rb') as f:
        f_batch2_x = pickle.load(f)
    with open(args.test_batch_unaligned, 'rb') as f:
        test_x = pickle.load(f)
    with open(args.test_batch_y, 'rb') as f:
        test_y = pickle.load(f)
    with open(args.test_batch_aligned, 'rb') as f:
        f_test_x = pickle.load(f)

    paired_batch_classification_test_no_prep(ref_x,
                                             ref_x,
                                             ref_y,
                                             batch2_x,
                                             batch2_y,
                                             f_batch2_x,
                                             test_x,
                                             test_y,
                                             f_test_x, args)
        

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('classificaiton-test', description='Run classification test on aligned data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
    parser.add_argument('ref_batch')
    parser.add_argument('ref_batch_y')
    parser.add_argument('batch2_unaligned')
    parser.add_argument('batch2_aligned')
    parser.add_argument('batch2_y')
    parser.add_argument('test_batch_unaligned')
    parser.add_argument('test_batch_aligned')
    parser.add_argument('test_batch_y')
    parser.add_argument('--input_normalization', help='Type of input normalizatio to apply.', choices=['l2', 'std', 'log', 'None'], default='l2')

    args = parser.parse_args()


    # test_with_pca(args)
    test_no_preprocessing(args)
