# import pdb; pdb.set_trace()
import argparse
import os
import pickle
import glob
from os.path import join
from collections import defaultdict
from pathlib import Path

import pandas as pd
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

def plot_lisi(df, alignment_task, output_folder):
    sort_order = SORT_ORDER
    df['ord'] = df.apply(lambda row: sort_order[row['method']], axis=1)
    df.sort_values('ord', inplace=True)
    sns.set(style="whitegrid")

    ax = sns.boxplot(x="score", y="method", hue="metric", data=df, palette="Set3", orient="h", showfliers=False, hue_order=[alignment_task.batch_key, alignment_task.ct_key])

    # groups = df.groupby(['method', 'metric'], sort=True)
    # means = groups['score'].mean()
    # medians = groups['score'].median()
    # q1 = groups['score'].quantile(0.25)
    # q2 = groups['score'].quantile(0.75)
    # iqr = q2.max() - q1.min()
    # whisker = q2.mean() + 1.5 * iqr
    # print('whisker: {}'.format(whisker))
    # stds = groups['score'].std()
    
    # print(groups)
    # print()
    # print(medians)
    # # new_left = df['score'].min()
    # max_point = df['score'].max()
    # print('max_point: {}'.format(max_point))

    # # new_right = max_point + (max_point - new_left) / 5

    # for tick, label in enumerate(ax.get_yticklabels()):
    #     print(tick)
    #     label = label.get_text()
    #     print(label)


    #     dataset_lisi_median = medians[label][alignment_task.batch_key]
    #     celltype_lisi_median = medians[label][alignment_task.ct_key]
    #     print(dataset_lisi_median)
    #     print(celltype_lisi_median)
    #     print()
    #     ax.text(whisker, tick - 0.19, '{:.3f}'.format(dataset_lisi_median), horizontalalignment='left', size='x-small', color='r', weight='semibold')
    #     ax.text(whisker, tick + 0.19, '{:.3f}'.format(celltype_lisi_median), horizontalalignment='left', size='x-small', color='r', weight='semibold')

    # print(new_left)
    # print(new_right)
    #ax.set_xlim(left=left, right=right)
    #plt.legend(bbox_to_anchor=(1.1, 1))
    #plt.legend(loc='upper left', bbox_to_anchor=(1,1))

    current_handles, current_labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    for l in current_labels:
        if l == 'protocol' or l == 'cell_age' or l == 'dataset':
            new_l = 'iLISI'
        else:
            new_l = 'cLISI'
        print(f'old: {l}, new: {new_l}')
        new_labels.append(new_l)
    plt.legend(current_handles, new_labels)
    ax.set_title('Scores on Task: {}'.format(alignment_task.as_plot_string()))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('compile-results', description='Combine LISI scores from multiple experiments into summarizing plots.')
    parser.add_argument('root_folder', help='Root folder to search for result files.')
    parser.add_argument('output_folder', help='Path of output folder (created if not exists) to store plots in.')

    args = parser.parse_args()
    lisi_folder = Path(args.output_folder) / 'LISI'
    # clf_folder = Path(args.output_folder) / 'classification'
    # kbet_folder = Path(args.output_folder) / 'kBET'
    for path in [args.output_folder, lisi_folder]:#, clf_folder, kbet_folder]:
        if not os.path.exists(path):
            os.makedirs(path)

    results_by_task = defaultdict(list)
    for filename in glob.iglob(join(args.root_folder, '**/results.pickle'), recursive=True):
        print(filename)
        with open(filename, 'rb') as f:
            result = pickle.load(f)
            results_by_task[str(result['alignment_task'])].append(result)
            
    for task, results in results_by_task.items():
        print(task)
        method = []
        metric = []
        score = []
        for r in results:
            method.extend([r['method']]*(2*r['lisi'].shape[0]))
            for col in r['lisi'].columns:
                metric.extend([col]*r['lisi'].shape[0])
                score.extend(r['lisi'][col])
        df = pd.DataFrame(data={'method': method, 'metric': metric, 'score': score})
        plot_lisi(df, results[0]['alignment_task'], lisi_folder)
        
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
