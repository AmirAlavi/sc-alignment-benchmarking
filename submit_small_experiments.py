import argparse
import os
import subprocess
from pathlib import Path

from dataset_info import batch_columns, celltype_columns, batches_available, celltypes_available, sources_targets_selected



def get_parser():
    parser = argparse.ArgumentParser('submit-experiments', description='Submit multiple alignment experiment jobs.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')

    parser.add_argument('--datasets', nargs='+', help='List of datasets to run all methods on.', required=True)
    parser.add_argument('-n', '--name', help='Experiment name (a valid name for a folder).', required=True)
    parser.add_argument('--email', help='Email to send slurm status to.', required=True)

    return parser

def get_method_info():
    methods_info = [
        {
            'name': 'None',
            'cmd': '--method None'
        },
        {
            'name': 'SeuratV3',
            'cmd': '--method SeuratV3'
        },
        {
            'name': 'ScAlign',
            'cmd': '--method ScAlign'
        },
        {
            'name': 'MNN',
            'cmd': '--method MNN'
        },
        {
            'name': 'closest',
            'cmd': '--method ICP_align  --matching_algo closest',
        },
        {
            'name': 'greedy,0.25,2',
            'cmd': '--method ICP_align  --matching_algo greedy --source_match_thresh 0.25 --target_match_limit 2',
        },
        {
            'name': 'greedy,0.5,2',
            'cmd': '--method ICP_align  --matching_algo greedy --source_match_thresh 0.5 --target_match_limit 2',
        },
        {
            'name': 'greedy,0.75,2',
            'cmd': '--method ICP_align  --matching_algo greedy --source_match_thresh 0.75 --target_match_limit 2',
        },
        {
            'name': 'greedy,0.5,1',
            'cmd': '--method ICP_align  --matching_algo greedy --source_match_thresh 0.5 --target_match_limit 1',
        },
        {
            'name': 'greedy,0.5,5',
            'cmd': '--method ICP_align  --matching_algo greedy --source_match_thresh 0.5 --target_match_limit 5',
        },
        {
            'name': 'hungarian,0.25',
            'cmd': '--method ICP_align  --matching_algo hungarian --source_match_thresh 0.25',
        },
        {
            'name': 'hungarian,0.5',
            'cmd': '--method ICP_align  --matching_algo hungarian --source_match_thresh 0.5',
        },
        {
            'name': 'hungarian,0.75',
            'cmd': '--method ICP_align  --matching_algo hungarian --source_match_thresh 0.75',
        },
    ]
    
    for info_dict in methods_info:
        info_dict['folder'] = info_dict['name'].replace(',', '_')
    return methods_info

def write_commands_to_file(cmd_list, path):
    with open(path, 'w') as f:
        for command_line in cmd_list:
            f.write(command_line + '\n')

if __name__ == '__main__':
    args = get_parser().parse_args()

    root = Path(args.name)
    os.makedirs(root)

    cpu_job_commands = []
    gpu_job_commands = []

    prefix = 'python alignment_experiment.py'
    # common_ICP_args = '--max_epochs=100 --max_steps=30  --plot_every_n 5 --input_space GENE  --do_kBET_test --filter_hvg --bias --tolerance 1e-4'
    # regularization = '--xentropy_loss_wt 0.1 --l2_reg 0'
    common_ICP_args = '--max_epochs=100 --max_steps=30  --plot_every_n 5 --input_space GENE  --do_kBET_test --filter_hvg --bias --tolerance 1e-4 --act=tanh'
    regularization = '--xentropy_loss_wt 1 --l2_reg 0'
    
    for ds in args.datasets:
        for source_target in sources_targets_selected[ds]:
            for method in get_method_info():
                method_path = method['folder']
                
                run_dir = root / f'{method_path}_{ds}_{source_target[0]}_{source_target[1]}'
                dataset_args = f'--dataset {ds} --source {source_target[0]} --target {source_target[1]}'
                cmd = f'{prefix} {method["name"]} {run_dir} {method["cmd"]} {regularization} {common_ICP_args} {dataset_args}'
                
                # cmd = 'python alignment_experiment.py -o {} --method {} --dataset {} --source {} --target {} --input_space {} --xentropy_loss_wt {} --source_match_thresh {} --l2_reg {}'.format(run_dir, method, ds, source_target[0], source_target[1], input_space, args.xentropy_loss_wt, args.source_match_thresh, args.l2_reg)
                if 'ICP' in method['cmd'] and ds != 'panc8':
                    gpu_job_commands.append(cmd)
                else:
                    cpu_job_commands.append(cmd)

                for ct in celltypes_available[ds]:
                    ct_run_dir = f'{run_dir}_{ct}'
                    ct_dataset_args = f'{dataset_args} --leaveOut {ct}'
                    ct_cmd = f'{prefix} {method["name"]} {ct_run_dir} {method["cmd"]} {regularization} {common_ICP_args} {ct_dataset_args}'
                    if 'ICP' in method['cmd']:
                        gpu_job_commands.append(ct_cmd)
                    else:
                        cpu_job_commands.append(ct_cmd)

    cpu_commands_file = root / Path('_tmp_cpu_commands_list.txt')
    gpu_commands_file = root / Path('_tmp_gpu_commands_list.txt')
    write_commands_to_file(cpu_job_commands, cpu_commands_file)
    write_commands_to_file(gpu_job_commands, gpu_commands_file)
    slurm_out = root / Path('slurm_cpu_array_%A_%a.out')
    slurm_err = root / Path('slurm_cpu_array_%A_%a.err')
    cpu_partitions = 'zbj1,zbj1-bigmem,pool1,pool3-bigmem'
    gpu_partitions = 'gpu,gpu2'
    cores = 2
    if 'panc8' in args.datasets:
        mem_per_core = '16G'
    else:
        mem_per_core = '8G'
    cpu_submit_cmd = f'sbatch --job-name cpu_align -p {cpu_partitions}  -n 1 -c {cores} --mem-per-cpu {mem_per_core} --array=0-{len(cpu_job_commands)-1} --mail-user {args.email} --mail-type FAIL --output {slurm_out} --error {slurm_err} singularity_execute.sh slurm_array.sh {cpu_commands_file}'



    print(cpu_submit_cmd)
    subprocess.run(cpu_submit_cmd.split())
    print()
    if len(gpu_job_commands) > 0:
        slurm_out = root / Path('slurm_gpu_array_%A_%a.out')
        slurm_err = root / Path('slurm_gpu_array_%A_%a.err')
        gpu_submit_cmd = f'sbatch --job-name gpu_align -p {gpu_partitions} --gres=gpu:1 -n 1 -c {cores} --mem-per-cpu {mem_per_core} --array=0-{len(gpu_job_commands)-1} --mail-user {args.email} --mail-type FAIL --output {slurm_out} --error {slurm_err} singularity_execute.sh slurm_array.sh {gpu_commands_file}'

        print(gpu_submit_cmd)
        subprocess.run(gpu_submit_cmd.split())

