batch_columns = {
    'Kowalcyzk': 'cell_age',
    'CellBench': 'protocol',
    'panc8': 'dataset',
    'scQuery_retina': 'accession',
    'scQuery_tcell': 'accession',
    'scQuery_lung': 'accession',
    'scQuery_pancreas': 'accession',
    'scQuery_ESC': 'accession',
    'scQuery_HSC': 'accession',
    'scQuery_combined': 'batch',
    'pbmcsca_low': 'protocol'
}

celltype_columns = {
    'Kowalcyzk': 'cell_type',
    'CellBench': 'cell_line_demuxlet',
    'panc8': 'celltype',
    'scQuery_retina': 'label_name',
    'scQuery_tcell': 'label_name',
    'scQuery_lung': 'label_name',
    'scQuery_pancreas': 'label_name',
    'scQuery_ESC': 'label_name',
    'scQuery_HSC': 'label_name',
    'scQuery_combined': 'label_name',
    'pbmcsca_low': 'CellType'
}

batches_available = {
    'Kowalcyzk': ['young', 'old'],
    'CellBench': ['10x', 'CELseq2', 'Dropseq'],
    'panc8': ['celseq', 'celseq2', 'smartseq2', 'fluidigmc1', 'indrop1', 'indrop2', 'indrop3', 'indrop4'],
    'pbmcsca_low': ['Smart-seq2', 'CEL-Seq2']
}

celltypes_available = {
    'Kowalcyzk': ["LT", "MPP", "ST"],
    'CellBench': ["H1975", "H2228", "HCC827"],
    'panc8': ["alpha", "beta", "ductal", "acinar"]
}

sources_targets_selected = {
    'Kowalcyzk': [('young', 'old')],
    'CellBench': [('Dropseq', 'CELseq2'), ('Dropseq', '10x'), ('CELseq2', '10x')],
    'panc8': [('celseq', 'fluidigmc1'), ('celseq', 'fluidigmc1'), ('celseq', 'celseq2'), ('celseq', 'smartseq2'), ('celseq', 'smartseq2'), ('indrop1', 'indrop2')]
}
