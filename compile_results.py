#import pdb; pdb.set_trace()
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

SORT_ORDER = {
    'None': 0,
    'MNN': 1,
    'SeuratV3': 2,
    'ScAlign': 3,
    'ICP2_xentropy': 4
}

def plot_lisi(lisi_dfs, method_names, alignment_task, output_folder):
    fig, (ilisi_ax, clisi_ax) = plt.subplots(1, 2)
    fig.suptitle(alignment_task)
    #sub_grid = figure_grid[i,j].subgridspec(1, 2, wspace=0.9)
    #ilisi_ax = plt.Subplot(fig, sub_grid[0])
    ilisi_ax.set_title('Dataset mixing')
    lisi_data = [df[alignment_task.batch_key].values for df in lisi_dfs]
    ilisi_ax.boxplot(lisi_data, vert=False, labels=method_names, showfliers=False)
    ilisi_ax.set_xlabel('iLISI')
    
    clisi_ax.set_title('Cell-type mixing')
    lisi_data = [df[alignment_task.ct_key] for df in lisi_dfs]
    clisi_ax.boxplot(lisi_data, vert=False, labels=method_names, showfliers=False)
    clisi_ax.set_xlabel('cLISI')
    plt.subplots_adjust(wspace=0.6)
    plt.savefig(output_folder / '{}.png'.format(alignment_task.as_path()))
    plt.savefig(output_folder / '{}.svg'.format(alignment_task.as_path()))
    plt.savefig(output_folder / '{}.pdf'.format(alignment_task.as_path()))
    plt.close()

def plot_clf(df, alignment_task, output_folder):
    df['ord'] = df.apply(lambda row: SORT_ORDER[row['method']], axis=1)
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

def plot_seaborn_lisi(df, alignment_task, output_folder, left=0.9, right=2.5):
    df['ord'] = df.apply(lambda row: SORT_ORDER[row['method']], axis=1)
    df.sort_values('ord', inplace=True)
    sns.set(style="whitegrid")

    ax = sns.boxplot(x="score", y="method", hue="metric", data=df, palette="Set3", orient="h", showfliers=False)

    groups = df.groupby(['method', 'metric'], sort=False)
    means = groups['score'].mean()
    stds = groups['score'].std()

    print(groups)
    print()
    print(means)
    print()
    print(stds)

    print(means.shape)
    print(stds.shape)

    annotes = [r'{:.2f}$\pm${:.3f}'.format(m, s) for m, s in zip(means, stds)]
    print(annotes)

    # new_left = df['score'].min()
    # max_point = df['score'].max()
    # new_right = max_point + (max_point - new_left) / 5

    pos = range(len(annotes))
    for tick,label in zip(pos, ax.get_yticklabels()):
        print(tick)
        print(label)
        print(annotes[(2*tick)])
        print(annotes[(2*tick) + 1])
        print()
        ax.text(right - 0.25, tick - 0.2, annotes[(2*tick)], horizontalalignment='left', size='x-small', color='r', weight='semibold')
        ax.text(right - 0.25, tick + 0.2, annotes[(2*tick) + 1], horizontalalignment='left', size='x-small', color='r', weight='semibold')

    # print(new_left)
    # print(new_right)
    ax.set_xlim(left=left, right=right)
    plt.legend(bbox_to_anchor=(1.1, 1))

    ax.set_title('Scores on Task: {}'.format(alignment_task.as_plot_string()))

    plt.savefig(output_folder / '{}_sns.png'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.savefig(output_folder / '{}_sns.svg'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.savefig(output_folder / '{}_sns.pdf'.format(alignment_task.as_path()), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('compile-results', description='Combine LISI scores from multiple experiments into summarizing plots.')
    parser.add_argument('root_folder', help='Root folder to search for result files.')
    parser.add_argument('output_folder', help='Path of output folder (created if not exists) to store plots in.')

    args = parser.parse_args()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    results_by_task = defaultdict(list)
    for filename in glob.iglob(join(args.root_folder, '**/results.pickle'), recursive=True):
        print(filename)
        with open(filename, 'rb') as f:
            result = pickle.load(f)
            results_by_task[str(result['alignment_task'])].append(result)
    for task, results in results_by_task.items():
        print(task)
        print(len(results))
        scores = [r['lisi'] for r in results]
        methods = [r['method'] for r in results]
        print(methods)
        plot_lisi(scores, methods, results[0]['alignment_task'], Path(args.output_folder))
        method = []
        metric = []
        score = []
        for r in results:
            method.extend([r['method']]*(2*r['lisi'].shape[0]))
            for col in r['lisi'].columns:
                metric.extend([col]*r['lisi'].shape[0])
                score.extend(r['lisi'][col])
        df = pd.DataFrame(data={'method': method, 'metric': metric, 'score': score})
        plot_seaborn_lisi(df, results[0]['alignment_task'], Path(args.output_folder))
        
    for task, results in results_by_task.items():
        method = []
        data = []
        acc = []
        # auc = []
        for r in results:
            method.extend([r['method']]*3)
            for dataset in ['target', 'source', 'source_aligned']:
                data.append(dataset)
                acc.append(r['clf']['{}_acc'.format(dataset)])
                # auc.append(r['clf']['{}_auc'.format(dataset)])
        # df = pd.DataFrame(data={'method': method, 'data': data, 'acc': acc, 'auc': auc})
        df = pd.DataFrame(data={'method': method, 'data': data, 'acc': acc})
        plot_clf(df, results[0]['alignment_task'], Path(args.output_folder))
    


