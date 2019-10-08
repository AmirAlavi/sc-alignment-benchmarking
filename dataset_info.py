batch_columns = {
    'Kowalcyzk': 'cell_age',
    'CellBench': 'protocol',
    'panc8': 'dataset'
}

celltype_columns = {
    'Kowalcyzk': 'cell_type',
    'CellBench': 'cell_line_demuxlet',
    'panc8': 'celltype'
}

batches_available = {
    'Kowalcyzk': ['young', 'old'],
    'CellBench': ['10x', 'CELseq2', 'Dropseq'],
    'panc8': ['celseq', 'celseq2', 'smartseq2', 'fluidigmc1', 'indrop1', 'indrop2', 'indrop3', 'indrop4']
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
