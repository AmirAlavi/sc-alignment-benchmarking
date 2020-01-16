import argparse
import math
import json
import sys
import pickle
#import pdb; pdb.set_trace()
from collections import defaultdict, namedtuple
from typing import List, Tuple
from os import makedirs
from os.path import exists, join

#sys.setrecursionlimit(5000)

import mygene
import numpy as np
import pandas as pd
from scipy.spatial.distance import jaccard
import anndata

TREE_FILE = 'trees.pickle'

def get_parser():
    parser = argparse.ArgumentParser('scquery-data-prep', description='Load and label the scQuery dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
    parser.add_argument('expression_h5_file', help='Path to scQuery hdf5 data file.')
    parser.add_argument('ontology_mappings_json_file', help='Path to json file which contains mappings of cell_IDs to lists of matched cell types.')
    #parser.add_argument('alignment_metadata_file', help='Path to file containing alignment metadata.')
    parser.add_argument('out_file', help='Path of resulting h5ad to write.')
    return parser

def get_accessions_from_accessionSeries(accessionSeries_list):
    accessions = []
    for accessionSeries in accessionSeries_list:
        accession = accessionSeries.split('_')[0]
        accessions.append(accession)
    return np.array(accessions)


def convert_entrez_to_symbol(entrezIDs):
    mg = mygene.MyGeneInfo()
    result = mg.getgenes(entrezIDs, fields='symbol', species='mouse')
    symbols = [d['symbol'].lower() for d in result]
    return symbols


def convert_rpkm_to_counts(rpkm_df, args):
    print('Converting to counts...')
    with open('trees.pickle', 'rb') as f:
        trees = pickle.load(f)
    gene_lengths = trees['gene_length'] # dict of geneID -> length in KB
    gene_lengths.index = gene_lengths.index.map(int) # necessary because the index (geneIDs) was strings originally
    
    with pd.HDFStore(args.expression_h5_file) as store:
        alignment_meta = store['alignment_metadata']
    read_counts = alignment_meta['read_count'].loc[rpkm_df.index] # select only the values for the samples present in our rpkm matrix

    # Sanity check that the orderings are what we expect
    np.testing.assert_array_equal(rpkm_df.index, read_counts.index)
    rpkm_df.columns = rpkm_df.columns.astype('int')
    np.testing.assert_array_equal(rpkm_df.columns, gene_lengths.index)
    print("passed assertions")
    counts_mat = rpkm_df
    counts_mat = counts_mat.mul(gene_lengths, axis=1)
    counts_mat = counts_mat.mul(read_counts, axis=0)
    counts_mat /= 1000000
    counts_mat = counts_mat.round()
    counts_mat = counts_mat.astype(int)
    print(counts_mat.shape)
    return counts_mat

def assign_unique_terms(mappings, rpkm_df):
    selected_cells = []
    selected_labels = []
    for key, value in mappings.items():
        if len(value) == 1:
            selected_cells.append(key)
            selected_labels.append(value[0])
    #rpkm_df.drop_duplicates(inplace=True)
    expression_vectors = rpkm_df.loc[selected_cells]
    counts = convert_rpkm_to_counts(expression_vectors, args)
    labels = pd.DataFrame(data=selected_labels, index=counts.index)
    adata = anndata.AnnData(X=counts)
    adata.obs['label'] = labels
    adata.obs['label_ID'] = adata.obs.apply(lambda x: x['label'].split(' ')[0], axis=1)
    adata.obs['label_name'] = adata.obs.apply(lambda x: ' '.join(x['label'].split(' ')[1:]), axis=1)
    gene_symbols = convert_entrez_to_symbol(counts.columns)
    adata.var['entrez'] = counts.columns
    adata.var['symbol'] = gene_symbols
    print("\nThe selected labels and their counts (no overlap):")
    unique_labels, label_counts = np.unique(selected_labels, return_counts=True)
    for l, c in zip(unique_labels, label_counts):
        print("\t", l, "\t", c)
    return adata

def load_cell_to_ontology_mapping(cells, ontology_mapping):
    empty_count = 0
    mappings = {}
    for cell in cells:
        if cell not in ontology_mapping:
            empty_count += 1
            continue
        terms_for_cell = ontology_mapping[cell]
        if len(terms_for_cell) == 0:
            empty_count += 1
        mappings[cell] = terms_for_cell
    print("Num cells with empty mappings: ", empty_count)
    return mappings


def analyze_cell_to_ontology_mapping(mappings):
    num_terms_mapped_to_l = []
    term_counts_d = defaultdict(int)
    for terms_for_cell in mappings.values():
        num_terms_mapped_to_l.append(len(terms_for_cell))
        for term in terms_for_cell:
            term_counts_d[term] += 1
    print("\nBincount of number of mapped terms for each cell:")
    print(np.bincount(num_terms_mapped_to_l))
    print("\nSorted list of terms by number of cells mapping to them (may overlap):")
    sorted_terms = sorted(term_counts_d.items(), key=lambda item: item[1], reverse=True)
    for term in sorted_terms:
        print(term)
    return term_counts_d


def filter_cell_to_ontology_terms(mappings, term_counts_d):
    terms_to_ignore = set()
    for term, count in term_counts_d.items():
        if count < 75 or 'NCBITaxon' in term or 'PR:' in term or 'PATO:' in term or 'GO:' in term or 'CLO:' in term:
            terms_to_ignore.add(term)
    # Terms that just don't seem that useful, or had too much overlap with another term that was more useful
    terms_to_ignore.add('UBERON:0000006 islet of Langerhans')
    terms_to_ignore.add('CL:0000639 basophil cell of pars distalis of adenohypophysis')
    terms_to_ignore.add('CL:0000557 granulocyte monocyte progenitor cell')
    terms_to_ignore.add('UBERON:0001068 skin of back')
    terms_to_ignore.add('CL:0000034 stem cell')
    terms_to_ignore.add('CL:0000048 multi fate stem cell')
    terms_to_ignore.add('UBERON:0000178 blood')
    terms_to_ignore.add('UBERON:0001135 smooth muscle tissue')
    terms_to_ignore.add('UBERON:0001630 muscle organ')
    terms_to_ignore.add('CL:0000000 cell')
    terms_to_ignore.add('CL:0000080 circulating cell')

    # Clean the mappings
    for cell in mappings.keys():
        terms = mappings[cell]
        mappings[cell] = [term for term in terms if term not in terms_to_ignore]
    return mappings


if __name__ == '__main__':
    args = get_parser().parse_args()
    # Open the hdf5 file that needs to be prepped (supplied as argument to this script)
    h5_store = pd.HDFStore(args.expression_h5_file)
    print("loaded h5 file")
    rpkm_df = h5_store['rpkm']
    h5_store.close()
    print(rpkm_df.shape)
    if 'GSE61300_GSM1501785' in rpkm_df.index:
        rpkm_df.drop('GSE61300_GSM1501785', inplace=True) # hack because drop_duplicates doesn't work
    if 'GSE61300_GSM1501786' in rpkm_df.index:
        rpkm_df.drop('GSE61300_GSM1501786', inplace=True)
    rpkm_df.fillna(0, inplace=True)
    with open(args.ontology_mappings_json_file, 'r') as f:
        cell_to_terms = json.load(f)
    mappings = load_cell_to_ontology_mapping(rpkm_df.index, cell_to_terms)

    # Filter down
    # Analyze the mappings
    print("\n\nBEFORE FILTERING")
    term_counts_d = analyze_cell_to_ontology_mapping(mappings)
    # Filter the mappings
    mappings = filter_cell_to_ontology_terms(mappings, term_counts_d)
    print("\n\nAFTER FILTERING")
    analyze_cell_to_ontology_mapping(mappings)
    
    adata = assign_unique_terms(mappings, rpkm_df)
    accessions = get_accessions_from_accessionSeries(adata.obs_names)
    adata.obs['accession'] = accessions

    adata.write(filename=args.out_file, compression='gzip')
        
    print("done")

