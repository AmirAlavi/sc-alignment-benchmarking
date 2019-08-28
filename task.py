class AlignmentTask(object):
    """Specifies the details of the task of aligning source data to a target space.

    Almost just a Plain-old-data class but with some pretty printing functions.

    Args:
        ds_key (str): 
        batch_key (str):
        ct_key (str):
        source_batch (str):
        target_batch (str):
        leave_out_ct (str, optional): Leave one cell type out of the target set.
    """
    def __init__(self, ds_key, batch_key, ct_key, source_batch, target_batch, leave_out_ct=None):
        self.ds_key = ds_key # dataset key
        self.batch_key = batch_key
        self.ct_key = ct_key # cell type key
        self.source_batch = source_batch
        self.target_batch = target_batch
        self.leave_out_ct = leave_out_ct
    
    def as_title(self):
        if self.leave_out_ct is not None:
            return '{}:\n{}->{}\n(\\{})'.format(self.ds_key, self.source_batch, self.target_batch, self.leave_out_ct)
        else:
            return '{}:\n{}->{}'.format(self.ds_key, self.source_batch, self.target_batch)
        
    def __str__(self):
        return self.as_title().replace('\n', '')

    def as_path(self):
        if self.leave_out_ct is not None:
            return '{}-{}_to_{}(out {})'.format(self.ds_key, self.source_batch, self.target_batch, self.leave_out_ct)
        else:
            return '{}-{}_to_{}'.format(self.ds_key, self.source_batch, self.target_batch)

def get_source_target(datasets, task_info, use_PCA=False):
    """Get the source data to be projected, as well as the target data on which it will be projected,
    both as np.ndarrays.
    """
    source_idx = datasets[task_info.ds_key].obs[task_info.batch_key] == task_info.source_batch
    if task_info.leave_out_ct is not None:
        target_idx = (datasets[task_info.ds_key].obs[task_info.batch_key] == task_info.target_batch) & (datasets[task_info.ds_key].obs[task_info.ct_key] != task_info.leave_out_ct)
    else:
        target_idx = datasets[task_info.ds_key].obs[task_info.batch_key] == task_info.target_batch
    if use_PCA:
        source = datasets[task_info.ds_key].obsm['PCA'][source_idx]
        target = datasets[task_info.ds_key].obsm['PCA'][target_idx]
    else:
        source = datasets[task_info.ds_key].X[source_idx]
        target = datasets[task_info.ds_key].X[target_idx]
    # A dict of <cell_type, indices where it occurs in concatenated (source, target) vector of cells>
    type_index_dict = {}
    combined_types = np.concatenate((datasets[task_info.ds_key].obs[task_info.cell_type_key][source_idx],
                                     datasets[task_info.ds_key].obs[task_info.cell_type_key][target_idx]))
    for cell_type in np.unique(combined_types):
        type_index_dict[cell_type] = np.where(combined_types == cell_type)[0]
    subset_meta = pd.concat((datasets[task_info.ds_key].obs[source_idx], datasets[task_info.ds_key].obs[target_idx]), axis=0)
    return source, target, type_index_dict, subset_meta