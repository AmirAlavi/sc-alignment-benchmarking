import argparse


def get_parser():
    parser = argparse.ArgumentParser('align-experiment', description='Run benchmarking on methods and datasets for scRNA-Seq dataset batch alignment (data integration)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')

    parser.add_argument('--methods', nargs='+', help='List of methods to run.', required=True)
    parser.add_argument('--datasets', nargs='+', help='List of datasets to run all methods on.', required=True)
    parser.add_argument('-n', '--name', help='Experiment name (a valid name for a folder).')
    parser.add_argument('--no_standardize', help='Do not StandardScale the input data.', action='store_true')
    parser.add_argument('--n_PC', help='Number of Principle Components of data to use.', type=int, default=100)
    parser.add_argument('--input_space', help='Which data input space to use.', choices=['GENE', 'PCA'], default='PCA')
    

    cellbench = parser.add_argument_group('CellBench Options')
    cellbench.add_argument('--CellBenchSource', help='Source batch for CellBench data.', default='Dropseq')
    cellbench.add_argument('--CellBenchTarget', help='Target batch for CellBench data.', default='CELseq2')
    cellbench.add_argument('--CellBenchLeaveOut', nargs='*', help='Leave-out cell types for CellBench data.', default=['H1975', 'H2228', 'HCC827'])

    kowal = parser.add_argument_group('Kowalcyzk Options')
    kowal.add_argument('--KowalSource', help='Source batch for Kowalcyzk data.', default='young')
    kowal.add_argument('--KowalTarget', help='Target batch for Kowalcyzk data.', default='old')
    kowal.add_argument('--KowalLeaveOut', nargs='*', help='Leave-out cell types for Kowalcyzk data.', default=['LT', 'MPP', 'ST'])

    panc8 = parser.add_argument_group('panc8 Options')
    panc8.add_argument('--panc8Source', help='Source batch for panc8 data.', default='celseq')
    panc8.add_argument('--panc8Target', help='Target batch for panc8 data.', default='celseq2')
    panc8.add_argument('--panc8LeaveOut', nargs='*', help='Leave-out cell types for panc8 data.', default=['alpha', 'beta'])

    icp = parser.add_argument_group('ICP options')
    icp.add_argument('--source_match_thresh', help='Portion of source points that need to be matched to a target point', type=float, default=0.5)
    icp.add_argument('--epochs', help='Number of iterations to run fitting (training).', type=int, default=100)
    icp.add_argument('--xentropy_loss_wt', help='For ICP + xentropy, the weight of the xentropy penalty', type=float, default=10)
    icp.add_argument('--nlayers', help='Number of layers in neural network data transformer.', type=int, choices=[1, 2], default=1)
    icp.add_argument('--act', help='Activation function to use in neural network (only for 2 layer nets).', )
    icp.add_argument('--bias', help='Use bias term in neural nets.', action='store_true')
    icp.add_argument('--lr', help='Learning rate in fitting.', type=float, default=1e-3)
    icp.add_argument('--plot_every_n', help='Plot the data using the neural net aligner every n steps.', type=int, default=5)
    
    scalign = parser.add_argument_group('ScAlign options')
    scalign.add_argument('--scalign_max_steps', help='Maximum epochs.', type=int, default=15000)
    scalign.add_argument('--scalign_batch_size', help='Batch size.', type=int, default=300)
    scalign.add_argument('--scalign_lr', help='Learning rate.', type=float, default=1e-4)
    scalign.add_argument('--scalign_architecture', help='Which pre-defined architecture to use.', choices=['large'], default='large')
    scalign.add_argument('--scalign_emb_size', help='Size of embedding.', type=int, default=32)

    seurat = parser.add_argument_group('Seurat options')
    seurat.add_argument('--seurat_env_path', help='Path to SeuratV3 R environment.', default='C:\\Users\\samir\\Anaconda3\\envs\\seuratV3')
    seurat.add_argument('--seurat_dims', help='Dimensionality of the dataset in alignment.', type=int, default=30)

    return parser


