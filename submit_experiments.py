import argparse
import os
import subprocess
from pathlib import Path

from dataset_info import batch_columns, celltype_columns, batches_available, celltypes_available

def get_parser():
    parser = argparse.ArgumentParser('submit-experiments', description='Submit multiple alignment experiment jobs.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')

    parser.add_argument('--methods', nargs='+', help='List of methods to run.', required=True)
    parser.add_argument('--datasets', nargs='+', help='List of datasets to run all methods on.', required=True)
    parser.add_argument('-n', '--name', help='Experiment name (a valid name for a folder).')
    #parser.add_argument('--no_standardize', help='Do not StandardScale the input data.', action='store_true')
    parser.add_argument('--n_PC', help='Number of Principle Components of data to use.', type=int, default=100)
    #parser.add_argument('--input_space', help='Which data input space to use.', choices=['GENE', 'PCA'], default='PCA')
    
    parser.add_argument('--partition', help='Slurm partition to use.', default='zbj1')
    parser.add_argument('--n_cpu', help='Number of CPUs per job.', type=int, default=2)
    parser.add_argument('--mem', help='Amount of memory (per cpu)', default='16G')
    parser.add_argument('--email', help='Email to send slurm status to.', required=True)

    # cellbench = parser.add_argument_group('CellBench Options')
    # cellbench.add_argument('--CellBenchSource', help='Source batch for CellBench data.', default='Dropseq')
    # cellbench.add_argument('--CellBenchTarget', help='Target batch for CellBench data.', default='CELseq2')
    # cellbench.add_argument('--CellBenchLeaveOut', nargs='*', help='Leave-out cell types for CellBench data.', default=['H1975', 'H2228', 'HCC827'])

    # kowal = parser.add_argument_group('Kowalcyzk Options')
    # kowal.add_argument('--KowalSource', help='Source batch for Kowalcyzk data.', default='young')
    # kowal.add_argument('--KowalTarget', help='Target batch for Kowalcyzk data.', default='old')
    # kowal.add_argument('--KowalLeaveOut', nargs='*', help='Leave-out cell types for Kowalcyzk data.', default=['LT', 'MPP', 'ST'])

    # panc8 = parser.add_argument_group('panc8 Options')
    # panc8.add_argument('--panc8Source', help='Source batch for panc8 data.', default='celseq')
    # panc8.add_argument('--panc8Target', help='Target batch for panc8 data.', default='celseq2')
    # panc8.add_argument('--panc8LeaveOut', nargs='*', help='Leave-out cell types for panc8 data.', default=['alpha', 'beta'])

    # icp = parser.add_argument_group('ICP options')
    # icp.add_argument('--source_match_thresh', help='Portion of source points that need to be matched to a target point', type=float, default=0.5)
    # icp.add_argument('--epochs', help='Number of iterations to run fitting (training).', type=int, default=100)
    # icp.add_argument('--xentropy_loss_wt', help='For ICP + xentropy, the weight of the xentropy penalty', type=float, default=10)
    # icp.add_argument('--nlayers', help='Number of layers in neural network data transformer.', type=int, choices=[1, 2], default=1)
    # icp.add_argument('--act', help='Activation function to use in neural network (only for 2 layer nets).', )
    # icp.add_argument('--bias', help='Use bias term in neural nets.', action='store_true')
    # icp.add_argument('--lr', help='Learning rate in fitting.', type=float, default=1e-3)
    # icp.add_argument('--plot_every_n', help='Plot the data using the neural net aligner every n steps.', type=int, default=5)
    
    # scalign = parser.add_argument_group('ScAlign options')
    # scalign.add_argument('--scalign_max_steps', help='Maximum epochs.', type=int, default=15000)
    # scalign.add_argument('--scalign_batch_size', help='Batch size.', type=int, default=300)
    # scalign.add_argument('--scalign_lr', help='Learning rate.', type=float, default=1e-4)
    # scalign.add_argument('--scalign_architecture', help='Which pre-defined architecture to use.', choices=['large'], default='large')
    # scalign.add_argument('--scalign_emb_size', help='Size of embedding.', type=int, default=32)

    # seurat = parser.add_argument_group('Seurat options')
    # seurat.add_argument('--seurat_env_path', help='Path to SeuratV3 R environment.', default='C:\\Users\\samir\\Anaconda3\\envs\\seuratV3')
    # seurat.add_argument('--seurat_dims', help='Dimensionality of the dataset in alignment.', type=int, default=30)

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()

    job_commands = []
    for method in args.methods:

        if 'Kowalcyzk' in args.datasets:
            job_commands.append('python alignment_experiment --method {} --dataset Kowalcyzk --source {} --target {}'.format(method, 'young', 'old'))
            for ct in celltypes_available['Kowalcyzk']:
                job_commands.append('python alignment_experiment --method {} --dataset Kowalcyzk --source {} --target {} --leave_out {}'.format(method, 'young', 'old', ct))

        if 'CellBench' in args.datasets:
            job_commands.append('python alignment_experiment --method {} --dataset CellBench --source {} --target {}'.format(method, 'Dropseq', 'CELseq2'))
            for ct in celltypes_available['CellBench']:
                job_commands.append('python alignment_experiment --method {} --dataset CellBench --source {} --target {} --leave_out {}'.format(method, 'Dropseq', 'CELseq2', ct))

        if 'panc8' in args.datasets:
            job_commands.append('python alignment_experiment --method {} --dataset panc8 --source {} --target {}'.format(method, 'celseq', 'fluidigmc1'))
            for ct in celltypes_available['panc8']:
                job_commands.append('python alignment_experiment --method {} --dataset panc8 --source {} --target {} --leave_out {}'.format(method, 'celseq', 'fluidigmc1', ct))

    root = Path(args.name)
    os.makedirs(root)
    commands_file = root / Path('_tmp_commands_list.txt')
    with open(commands_file, 'w') as f:
        for command_line in job_commands:
            print(command_line)
            f.write(command_line + '\n')
    slurm_out = root / Path('scrna_train_array_%A_%a.out')
    slurm_err = root / Path('scrna_train_array_%A_%a.err')  
    submit_cmd = 'sbatch --job-name {} -p {} -n 1 -c {} --mem-per-cpu {} --array=0-{} --mail-user {} --mail-type FAIL --output {} --error {} slurm_array.sh {}'.format('align_exp', args.partition, args.n_cpu, args.mem, len(job_commands)-1, args.email, slurm_out, slurm_err, commands_file)
    print(submit_cmd)
    #subprocess.run(submit_cmd.split())


