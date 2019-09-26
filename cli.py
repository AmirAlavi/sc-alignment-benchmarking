import argparse


def get_parser():
    parser = argparse.ArgumentParser('align-experiment', description='Run benchmarking on methods and datasets for scRNA-Seq dataset batch alignment (data integration)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')

    parser.add_argument('--method', help='Which method to run.', required=True, choices=['None', 'ScAlign', 'MNN', 'SeuratV3', 'ICP', 'ICP2', 'ICP2_xentropy'])
    parser.add_argument('--dataset', help='Which dataset to run the alignment method on.', required=True, choices=['Kowalcyzk', 'CellBench', 'panc8'])
    parser.add_argument('-o', '--output_folder', help='Output folder for this experiment.')
    parser.add_argument('--no_standardize', help='Do not StandardScale the input data.', action='store_true')
    parser.add_argument('--n_PC', help='Number of Principle Components of data to use.', type=int, default=100)
    parser.add_argument('--input_space', help='Which data input space to use.', choices=['GENE', 'PCA'], default='PCA')
    

    cellbench = parser.add_argument_group('Alignment Task Options')
    cellbench.add_argument('--source', help='Source batch.', required=True)
    cellbench.add_argument('--target', help='Target batch.', required=True)
    cellbench.add_argument('--leaveOut', help='Leave-out cell type.')

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
    seurat.add_argument('--seurat_env_path', help='Path to SeuratV3 R environment.', default='C:\\Users\\Amir\\Anaconda3\\envs\\seuratV3')
    seurat.add_argument('--seurat_dims', help='Dimensionality of the dataset in alignment.', type=int, default=30)

    return parser


