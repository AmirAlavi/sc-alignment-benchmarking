import argparse


def get_parser():
    parser = argparse.ArgumentParser('align-experiment', description='Run benchmarking on methods and datasets for scRNA-Seq dataset batch alignment (data integration)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')

    parser.add_argument('name', help='Descriptive name of the method, usually some concatenation of the method and its specific parameters')
    parser.add_argument('output_folder', help='Output folder for this experiment.')
    parser.add_argument('--save_alignment', help='Save the resulting aligned cells to disk', action='store_true')

    parser.add_argument('--method', help='Which method to run.', required=True, choices=['None', 'ScAlign', 'MNN', 'SeuratV3', 'ICP', 'ICP2', 'ICP2_xentropy', 'ICP_converge', 'ICP2_xentropy_converge', 'ICP_align', 'ICP_rigid', 'ICP_affine', 'ICP_stacked_aes'])
    parser.add_argument('--dataset', help='Which dataset to run the alignment method on.', required=True, choices=['Kowalcyzk', 'CellBench', 'panc8', 'scQuery_retina', 'scQuery_tcell', 'scQuery_lung', 'scQuery_pancreas', 'scQuery_ESC', 'scQuery_HSC', 'scQuery_combined', 'pbmcsca_low', 'pbmcsca_high'])
    parser.add_argument('--panc8_n_cell_types', help='Number of the top most populous cell types to include.', type=int, default=5)
    parser.add_argument('--pbmcsca_high_n_cell_types', help='Number of the top most populous cell types to include.', type=int, default=3)
    parser.add_argument('--standardize', help='StandardScale the input to PCA.', action='store_true')
    parser.add_argument('--n_PC', help='Number of Principle Components of data to use.', type=int, default=100)
    parser.add_argument('--input_space', help='Which data input space to use.', choices=['GENE', 'PCA'], default='PCA')
    parser.add_argument('--do_DE_test', help='Do a differential expression test with GO enrichment analysis using the aligned data.', action='store_true')
    parser.add_argument('--do_kBET_test', help='Compute the k-BET test to quantify batch correction, as a metric.', action='store_true')
    parser.add_argument('--kBET_env_path', help='Path to k-BET R environment.', default='/home/aalavi/anaconda2/envs/kBET')
    parser.add_argument('--do_clf_score', help='Do check of improved classifier accuracy.', action='store_true')
    parser.add_argument('--do_clf_test', help='Do check of improved classifier accuracy.', action='store_true')
    parser.add_argument('--clf_test_batch', help='Test batch for the "do_clf_test" option.')
    

    cellbench = parser.add_argument_group('Alignment Task Options')
    cellbench.add_argument('--source', help='Source batch.', required=True)
    cellbench.add_argument('--target', help='Target batch.', required=True)
    cellbench.add_argument('--leaveOut', help='Leave-out cell type.')
    cellbench.add_argument('--leaveOutSource', help='Cell type to hide in source set during fitting')

    icp = parser.add_argument_group('ICP options')
    icp.add_argument('--matching_algo', help='Which matching algorithm to use to pair up source points to target points.', choices=['closest', 'greedy', 'hungarian', 'mnn'], default='greedy')
    icp.add_argument('--source_match_thresh', help='Portion of source points that need to be matched to a target point', type=float, default=0.5)
    icp.add_argument('--target_match_limit', help='Maximum number of times a target point can be assigned to.', type=int, default=2)
    icp.add_argument('--min_steps', help='Min number of steps to run iterative point-cloud registration algorithm.', type=int, default=100)
    icp.add_argument('--max_steps', help='Number of steps to run iterative point-cloud registration algorithm.', type=int, default=100)
    icp.add_argument('--tolerance', help='Stopping criterion for algorithm, if norm of difference in transformed data between iterations is less than this, then stop.', type=float, default=0.25)
    icp.add_argument('--patience', help='Stopping criterion for algorithm, if no improvement in MSE distance of matched points for this many number of steps, then stop.', type=int, default=5)
    icp.add_argument('--max_epochs', help='Number of iterations to run fitting for affine transformation.', type=int, default=10000)
    icp.add_argument('--input_normalization', help='Type of input normalizatio to apply.', choices=['l2', 'std', 'log', 'None'], default='None')
    icp.add_argument('--mini_batching', help='Enable batched optimization.', action='store_true')
    icp.add_argument('--sparse', help='Enable sparse training.', action='store_true')
    icp.add_argument('--cpu_only', help='Only use CPU even if GPU is available', action='store_true')
    icp.add_argument('--batch_size', help='Mini batch size (if mini_batching enabled).', type=int, default=32)
    icp.add_argument('--subsample', help='Enable subsampling of large data for fast training.', action='store_true')
    icp.add_argument('--n_subsample', help='If subsampling, number of samples to take from each of source set and target set.', type=int, default=500)
    icp.add_argument('--xentropy_loss_wt', help='For ICP + xentropy, the weight of the xentropy penalty', type=float, default=10)
    icp.add_argument('--l2_reg', help='L2 regularization weight.', type=float, default=0.)
    icp.add_argument('--nlayers', help='Number of layers in neural network data transformer.', type=int, choices=[1, 2, 3, 5], default=1)
    icp.add_argument('--stacked_aes_sizes', help='Sizes of hidden layers for each autoencoder in stacked AEs transformer', nargs='+', type=int, default=64)
    icp.add_argument('--dropout', help='Amount of dropout, if any, to use betwen network layers.', type=float, default=0.)
    icp.add_argument('--batch_norm', help='Use batch norm layers betwen network layers.', action='store_true')
    icp.add_argument('--use_autoencoder', help='Use an autoencoder neural network to align the data.', action='store_true')
    icp.add_argument('--last_layer_linear', help="In autoencoders, don't apply a non-linearity after the last layer.", action='store_true')
    icp.add_argument('--enforce_pos', help='Last layer of transformer is ReLU to force output to be positive.', action='store_true')
    icp.add_argument('--act', help='Activation function to use in neural network (only for 2 layer nets).', )
    icp.add_argument('--bias', help='Use bias term in neural nets.', action='store_true')
    icp.add_argument('--lr', help='Learning rate in fitting.', type=float, default=1e-3)
    icp.add_argument('--opt', help='Optimizer to use for fitting transform.', choices=['sgd', 'adam'], default='adam')
    icp.add_argument('--plot_every_n', help='Plot the data using the neural net aligner every n steps.', type=int, default=5)
    icp.add_argument('--filter_hvg', help='Filter down to only highly variable genes.', action='store_true')
    icp.add_argument('--filter_hvg2', help='Filter down to only highly variable genes, but use twice as many of them as --filter_hvg (when sorted by variability, pick twice as many).', action='store_true')
    icp.add_argument('--filter_hvg_random', help='Filter down to only highly variable genes, and also include an equal amount of other randomly selected genes.', action='store_true')
    
    scalign = parser.add_argument_group('ScAlign options')
    scalign.add_argument('--scalign_max_steps', help='Maximum epochs.', type=int, default=15000)
    scalign.add_argument('--scalign_batch_size', help='Batch size.', type=int, default=300)
    scalign.add_argument('--scalign_lr', help='Learning rate.', type=float, default=1e-4)
    scalign.add_argument('--scalign_architecture', help='Which pre-defined architecture to use.', choices=['large'], default='large')
    scalign.add_argument('--scalign_emb_size', help='Size of embedding.', type=int, default=32)

    seurat = parser.add_argument_group('Seurat options')
    #seurat.add_argument('--seurat_env_path', help='Path to SeuratV3 R environment.', default='C:\\Users\\Amir\\Anaconda3\\envs\\seuratV3')
    seurat.add_argument('--seurat_env_path', help='Path to SeuratV3 R environment.', default='/home/aalavi/anaconda2/envs/seuratV3')
    seurat.add_argument('--seurat_dims', help='Dimensionality of the dataset in alignment.', type=int, default=30)

    return parser


