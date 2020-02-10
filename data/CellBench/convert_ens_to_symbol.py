#import pdb; pdb.set_trace()

import pandas as pd
import mygene
import sys

def convert_ens_to_symbol(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    print(df.shape)
    print(df.index)
    mg = mygene.MyGeneInfo()
    result = mg.getgenes(df.index, filds='symbol', species=sys.argv[2])
    not_found = [d['query'] for d in result if 'notfound' in d and d['notfound']]
    print('num not found: {}'.format(len(not_found)))
    found = {}
    for d in result:
        if 'symbol' in d:
            found[d['query']] = d['symbol']

    df.drop(not_found, inplace=True)
    df.rename(index=found, inplace=True)
    print(df.shape)
    print(df.index)
    df.to_csv(csv_path)

if __name__ == '__main__':
    convert_ens_to_symbol(sys.argv[1])

