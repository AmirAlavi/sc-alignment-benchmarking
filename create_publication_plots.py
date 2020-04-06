import pdb; pdb.set_trace()
import argparse
import os
import pickle
import glob
from os.path import join
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from submit_small_experiments import get_method_info

SORT_ORDER = {
    'None': 0,
    'MNN': 1,
    'SeuratV3': 2,
    'ScAlign': 3,
    'ICP-align': 4,
    'ICP-affine-greedy': 5,
    'ICP-affine-mnn': 6
    # 'greedy_thresh_0.25_limit_02': 5,
    # 'hungarian_thresh_0.25': 6,
    # 'greedy_thresh_0.50_limit_02': 7,
    # 'hungarian_thresh_0.50': 8,
    # 'greedy_thresh_0.75_limit_02': 9,
    # 'hungarian_thresh_0.75': 10,
    # 'greedy_thresh_0.50_limit_01': 11,
    # 'greedy_thresh_0.50_limit_05': 12
}

def get_sort_order():
    method_list = get_method_info()
    order = {}
    for i, method in enumerate(method_list):
        order[method['name']] = i
    return order

def plot_clf(df, alignment_task, output_folder):
    sort_order = get_sort_order()
    df['ord'] = df.apply(lambda row: sort_order[row['method']], axis=1)
    df.sort_values('ord', inplace=True)
    sns.set(style="whitegrid")
    ax = sns.barplot(x='data', y='acc', hue='method', data=df)
    ax.set_title('Classifier Accuracy on: {}'.format(alignment_task.as_plot_string()))

    plt.savefig(output_folder / '{}_acc.png'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.savefig(output_folder / '{}_acc.svg'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.savefig(output_folder / '{}_acc.pdf'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.close()

    # ax = sns.barplot(x='data', y='auc', hue='method', data=df)
    # ax.set_title('Classifier AUC on: {}'.format(alignment_task.as_plot_string()))

    # plt.savefig(output_folder / '{}_auc.png'.format(alignment_task.as_path()), bbox_inches='tight')
    # plt.savefig(output_folder / '{}_auc.svg'.format(alignment_task.as_path()), bbox_inches='tight')
    # plt.savefig(output_folder / '{}_auc.pdf'.format(alignment_task.as_path()), bbox_inches='tight')
    # plt.close()

def get_sort_order_by_score(df, alignment_task):
    sort_df = df.copy()
    def negate_iLISI(row):
        if row['metric'] == alignment_task.ct_key:
            return -row['score']
        else:
            return row['score']
    sort_df['score'] = df.apply(negate_iLISI, axis=1)
    sort_df = sort_df.groupby(['method', 'metric']).median().groupby('method').sum().sort_values(by='score', ascending=False)
    sort_df['order'] = (-sort_df['score']).argsort()
    sort_dict = {}
    for k, v in sort_df.to_dict(orient='index').items():
        sort_dict[k] = v['order']
    return sort_dict

def plot_lisi(df, alignment_task, output_folder):
    # sort_order = SORT_ORDER
    sort_order = get_sort_order_by_score(df, alignment_task)
    df['ord'] = df.apply(lambda row: sort_order[row['method']], axis=1)
    df.sort_values('ord', inplace=True)
    sns.set(style="whitegrid")

    ax = sns.boxplot(x="score", y="method", hue="metric", data=df, palette="Set2", orient="h", showfliers=False, hue_order=[alignment_task.batch_key, alignment_task.ct_key])

    current_handles, current_labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    for l in current_labels:
        if l == 'protocol' or l == 'cell_age' or l == 'dataset' or l == 'batch':
            new_l = 'iLISI'
        else:
            new_l = 'cLISI'
        print(f'old: {l}, new: {new_l}')
        new_labels.append(new_l)
    plt.legend(current_handles, new_labels)
    ax.set_title('Scores on {}'.format(alignment_task.as_plot_string()))

    plt.savefig(output_folder / '{}_sns.png'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.savefig(output_folder / '{}_sns.svg'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.savefig(output_folder / '{}_sns.pdf'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.close()

def plot_kBET(df, alignment_task, output_folder):
    sort_order = get_sort_order()
    df['ord'] = df.apply(lambda row: sort_order[row['method']], axis=1)
    df.sort_values('ord', inplace=True)
    sns.set(style="whitegrid")

    ax = sns.boxplot(x="value", y="method", hue="metric", data=df, showfliers=True, hue_order=['kBET.expected', 'kBET.observed', 'kBET.signif'])

    plt.legend(loc='upper left', bbox_to_anchor=(1,1))

    ax.set_title('kBET Rejection Rates on Task: {}'.format(alignment_task.as_plot_string()))

    plt.savefig(output_folder / '{}_kbet.png'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.savefig(output_folder / '{}_kbet.svg'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.savefig(output_folder / '{}_kbet.pdf'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.close()

def rename_method(method, renames):
    for rn in renames:
        original, new = rn.split(',')
        if method == original:
            return new
    return method

def rename_methods(methods, renames):
    for i in range(len(methods)):
        methods[i] = rename_method(methods[i], renames)
    return methods

def rename_dataset(alignment_task, renames):
    for rn in renames:
        original, new = rn.split(',')
        if alignment_task.ds_key == original:
            alignment_task.ds_key = new
    return alignment_task

def load_LISI_df(args):
    method = []
    metric = []
    score = []
    dataset = []
    source = []
    target = []
    leaveOut = []
    task_plot_string = []
    task_ct_key = []
    task_batch_key = []
    for filename in glob.iglob(join(args.root_folder, '**/results.pickle'), recursive=True):
        print(filename)
        with open(filename, 'rb') as f:
            r = pickle.load(f)
        n_rows = 2*r['lisi'].shape[0]
        cur_method = r['method']
        if args.rename_method is not None:
            cur_method = rename_method(cur_method, args.rename_method)
        method.extend([cur_method]*(n_rows))
        for col in r['lisi'].columns:
            metric.extend([col]*r['lisi'].shape[0])
            score.extend(r['lisi'][col])
        
        cur_task = r['alignment_task']
        if args.rename_dataset is not None:
            cur_task = rename_dataset(cur_task, args.rename_dataset)
        dataset.extend([cur_task.ds_key] * n_rows)
        source.extend([cur_task.source_batch] * n_rows)
        target.extend([cur_task.target_batch] * n_rows)
        leaveOut.extend([cur_task.leave_out_ct] * n_rows)
        task_plot_string.extend([cur_task.as_plot_string()] * n_rows)
        task_ct_key.extend([cur_task.ct_key] * n_rows)
        task_batch_key.extend([cur_task.batch_key] * n_rows)
    df = pd.DataFrame(data={
        'method': method,
        'metric': metric,
        'score': score,
        'dataset': dataset,
        'source': source,
        'target': target,
        'leaveOut': leaveOut,
        'task_plot_string': task_plot_string,
        'task_ct_key': task_ct_key,
        'task_batch_key': task_batch_key})
    return df

sources = {
    'CellBench': ['CELseq2', 'Dropseq'],
    'Pancreas': ['indrop1', 'indrop2', 'indrop4'],
    'PBMC': ['10x Chromium (v2) A', '10x Chromium (v2) B', '10x Chromium (v3)']
}

def get_sort_order_by_score_2(df):
    sort_df = df.copy()
    def negate_iLISI(row):
        if row['metric'] == row['task_ct_key']:
            return -row['score']
        else:
            return row['score']
    sort_df['score'] = df.apply(negate_iLISI, axis=1)
    sort_df = sort_df.groupby(['method', 'metric']).median().groupby('method').sum().sort_values(by='score', ascending=False)
    sort_df['order'] = (-sort_df['score']).argsort()
    sort_dict = {}
    for k, v in sort_df.to_dict(orient='index').items():
        sort_dict[k] = v['order']
    return sort_dict

def plot_lisi_ax(df, ax):
    # sort_order = SORT_ORDER
    if len(np.unique(df.task_plot_string)) != 1:
        raise RuntimeError('Detected more than one alignment task in LISI plot!')
    sort_order = get_sort_order_by_score_2(df)
    df['ord'] = df.apply(lambda row: sort_order[row['method']], axis=1)
    df.sort_values('ord', inplace=True)
    sns.set(style="whitegrid")

    batch_key = df.task_batch_key.iloc[0]
    ct_key = df.task_ct_key.iloc[0]
    sns.boxplot(x="score", y="method", hue="metric", data=df, palette="Set2",
        orient="h", showfliers=False,
        hue_order=[batch_key, ct_key], ax=ax)

    current_handles, current_labels = ax.get_legend_handles_labels()
    new_labels = []
    for l in current_labels:
        if l == 'protocol' or l == 'cell_age' or l == 'dataset' or l == 'batch':
            new_l = 'iLISI'
        else:
            new_l = 'cLISI'
        print(f'old: {l}, new: {new_l}')
        new_labels.append(new_l)
    ax.legend(current_handles, new_labels)
    plot_title = df.task_plot_string.iloc[0].split(':')[1].strip()
    plot_title = plot_title.replace('Chromium ', '')
    ax.set_title(f'{plot_title}')
    ax.set(xlabel='', ylabel='')

def plot_overall_LISI_fig(df, output_folder):
    df = df[pd.isnull(df.leaveOut)]
    dataset_order = {'CellBench': 0, 'Pancreas': 1, 'PBMC': 2}
    datasets = np.unique(df.dataset)
    datasets_places = [dataset_order[ds] for ds in datasets]
    sort_idx = np.argsort(datasets_places)
    datasets = datasets[sort_idx]
    n_rows = len(datasets)
    n_cols = max([len(sources[ds]) for ds in datasets])

    def_figsize = matplotlib.rcParams['figure.figsize']
    figsize = [s*1.5 for s in def_figsize]
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, gridspec_kw={'hspace': 0.6, 'wspace': 0.75}, figsize=figsize)
    for i, dataset in enumerate(datasets):
        for j, source in enumerate(sources[dataset]):
            print(f'{dataset}: {source}')
            df_subset = df[(df.dataset == dataset) & (df.source == source)]
            plot_lisi_ax(df_subset, axs[i, j])
            if j == 0:
                axs[i, j].set(ylabel=dataset)
            if i == len(datasets) - 1 and j == int(n_cols / 2):
                axs[i, j].set(xlabel='score')
            if i != 0 or j != 0:
                axs[i, j].get_legend().remove()
    plt.savefig(output_folder / 'overall_LISI.png', bbox_inches='tight')
    #plt.savefig(output_folder / '{}_sns.svg'.format(alignment_task.as_path()), bbox_inches='tight')
    #plt.savefig(output_folder / '{}_sns.pdf'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('compile-results', description='Combine LISI scores from multiple experiments into summarizing plots.')
    parser.add_argument('root_folder', help='Root folder to search for result files.')
    parser.add_argument('output_folder', help='Path of output folder (created if not exists) to store plots in.')
    parser.add_argument('--rename_method', help='Change the text name of a particular method to appear in the plots.', action='append')
    parser.add_argument('--rename_dataset', help='Change the text name of a particular dataset to appear in the plots.', action='append')

    args = parser.parse_args()
    lisi_folder = Path(args.output_folder) / 'LISI'
    # clf_folder = Path(args.output_folder) / 'classification'
    # kbet_folder = Path(args.output_folder) / 'kBET'
    for path in [args.output_folder, lisi_folder]:#, clf_folder, kbet_folder]:
        if not os.path.exists(path):
            os.makedirs(path)

    df = load_LISI_df(args)
    plot_overall_LISI_fig(df, lisi_folder)

    # results_by_task = defaultdict(list)
    # for filename in glob.iglob(join(args.root_folder, '**/results.pickle'), recursive=True):
    #     print(filename)
    #     with open(filename, 'rb') as f:
    #         result = pickle.load(f)
    #         results_by_task[str(result['alignment_task'])].append(result)
            
    # for task, results in results_by_task.items():
    #     print(task)
    #     method = []
    #     metric = []
    #     score = []
    #     for r in results:
    #         method.extend([r['method']]*(2*r['lisi'].shape[0]))
    #         for col in r['lisi'].columns:
    #             metric.extend([col]*r['lisi'].shape[0])
    #             score.extend(r['lisi'][col])
    #     if args.rename_method is not None:
    #         method = rename_methods(method, args.rename_method)
    #     alignment_task = results[0]['alignment_task']
    #     if args.rename_dataset is not None:
    #         alignment_task = rename_dataset(alignment_task, args.rename_dataset)
    #     df = pd.DataFrame(data={'method': method, 'metric': metric, 'score': score})
    #     plot_lisi(df, alignment_task, lisi_folder)
        
    # for task, results in results_by_task.items():
    #     method = []
    #     data = []
    #     acc = []
    #     # auc = []
    #     for r in results:
    #         if r['clf'] == None:
    #             continue
    #         method.extend([r['method']]*2)
    #         for dataset in ['source', 'source_aligned']:
    #             data.append(dataset)
    #             acc.append(r['clf']['{}_acc'.format(dataset)])
    #             # auc.append(r['clf']['{}_auc'.format(dataset)])
    #     # df = pd.DataFrame(data={'method': method, 'data': data, 'acc': acc, 'auc': auc})
    #     df = pd.DataFrame(data={'method': method, 'data': data, 'acc': acc})
    #     plot_clf(df, results[0]['alignment_task'], clf_folder)
        
    # for task, results in results_by_task.items():
    #     print(task)
    #     print(len(results))
    #     # scores = [r['kbet_stats'] for r in results]
    #     methods = [r['method'] for r in results]
    #     print(methods)
    #     method = []
    #     metric = []
    #     value = []
    #     for r in results:
    #         kbet = r['kbet_stats']
    #         method.extend([r['method']]*kbet.size)
    #         for col in kbet.columns:
    #             metric.extend([col]*kbet.shape[0])
    #             value.extend(kbet[col])
    #     df = pd.DataFrame(data={'method': method, 'metric': metric, 'value': value})
    #     plot_kBET(df, results[0]['alignment_task'], kbet_folder)
