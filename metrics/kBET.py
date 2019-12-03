import tempfile
import subprocess
import platform
from pathlib import Path

import numpy as np
import pandas as pd

def kBET(X, meta_data, batch_key, kBET_env_path):
    print(X.shape)

    # import os
    # tmp_dir = 'testing_kBET'
    # os.makedirs(tmp_dir, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        working_dir = Path(tmp_dir)
        print("saving data for kBET")
        # Save aligned, integrated dataset
        kbet_x_file = working_dir / 'x.csv'
        np.savetxt(kbet_x_file, X, delimiter=',')
        kbet_batch_file = working_dir / 'batch.csv'
        np.savetxt(kbet_batch_file, meta_data[batch_key], delimiter=',', fmt='%s')
        kbet_result_file = working_dir / 'kbet_result.csv'
        # Run kBET
        #cmd = "C:\\Users\\samir\\Anaconda3\\envs\\seuratV3\\Scripts\\Rscript.exe  seurat_align.R {}".format(task.batch_key)
        kbet_env_path = Path(kBET_env_path)
        if platform.system() == 'Windows':
            bin_path = kbet_env_path / 'Library' / 'mingw-w64' / 'bin'
            rscript_path = kbet_env_path / 'Scripts' / 'Rscript.exe'
            cmd = f'set PATH={bin_path};%PATH% && {rscript_path} kbet_compute.R {kbet_x_file} {kbet_batch_file} {kbet_result_file}'
            cmd = cmd.split()
        else:
            bin_path = kbet_env_path / 'bin' 
            rscript_path = bin_path / 'Rscript'
            cmd = f'PATH="{bin_path}:$PATH" {rscript_path} kbet_compute.R {kbet_x_file} {kbet_batch_file} {kbet_result_file}'
        print('Running command: {}'.format(cmd))
        try:
            console_output = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            console_output = console_output.stdout.decode('UTF-8')
            print('Finished running')
            print(console_output)
        except subprocess.CalledProcessError as e:
            print("RUNNING kBET FAILED")
            print(e.stdout.decode('UTF-8'))
        kbet_stats = pd.read_csv(kbet_result_file, index_col=0)
        return kbet_stats
