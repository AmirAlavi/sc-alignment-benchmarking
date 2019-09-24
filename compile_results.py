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

