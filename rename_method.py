import argparse
import pickle
import glob
from os.path import join

if __name__ == '__main__':
    parser = argparse.ArgumentParser('rename-icp-method', description='rename method key in a folder')
    parser.add_argument('root_folder', help='Root folder to search for result files.')
    parser.add_argument('new_name', help='New method name to use')

    args = parser.parse_args()
    for filename in glob.iglob(join(args.root_folder, '**/results.pickle'), recursive=True):
        print(filename)
        with open(filename, 'rb') as f:
            result = pickle.load(f)
        if 'icp2' in filename:
            result['method'] = args.new_name
        with open(filename, 'wb') as f:
            pickle.dump(result, f)
    print('done')
