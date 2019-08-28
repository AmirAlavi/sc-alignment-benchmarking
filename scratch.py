# Temporary place for code

def before_and_after_plots(A, B, type_index_dict, aligner_fcn, standardize=True, do_B_transform=False):
    fig, axes = plt.subplots(2, 2, figsize=(20,20))
    # Before alignment
    if standardize:
        scaler = StandardScaler().fit(np.concatenate((A, B)))
        A = scaler.transform(A)
        B = scaler.transform(B)
    A_size = A.shape[0]
    combined = TSNE(n_components=2).fit_transform(np.concatenate((A, B)))
    axes[0,0].scatter(combined[:A_size,0], combined[:A_size,1], c='m', label='source', alpha=0.15)
    axes[0,0].scatter(combined[A_size:,0], combined[A_size:,1], c='b', label='target', alpha=0.15)
    axes[0,0].legend()
    axes[0,0].set_title('t-SNE (before)')
    for cell_type, idx in type_index_dict.items():
        axes[0,1].scatter(combined[idx, 0], combined[idx, 1], label=cell_type, alpha=0.15)
        axes[0,1].legend()
    # Aligned
    A = aligner_fcn(A)
    if do_B_transform:
        B = aligner_fcn(B)
    combined = TSNE(n_components=2).fit_transform(np.concatenate((A, B)))
    axes[1,0].scatter(combined[:A_size,0], combined[:A_size,1], c='m', label='source', alpha=0.15)
    axes[1,0].scatter(combined[A_size:,0], combined[A_size:,1], c='b', label='target', alpha=0.15)
    axes[1,0].legend()
    axes[1,0].set_title('t-SNE (after)')
    for cell_type, idx in type_index_dict.items():
        axes[1,1].scatter(combined[idx, 0], combined[idx, 1], label=cell_type, alpha=0.15)
        axes[1,1].legend()