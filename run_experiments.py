# Input directory must have the following structure:
# dir/
#   manifest/
#     manifest.txt # commands to run
#     completions.txt # successful completions
# Given a directory, looks for a manifest file (a txt file with running arguments on each line)
# loads the manifest
# checks for completed runs (via hash of command string)
# reports list of runs with checkmark next to them if already run
# starts running the ones not done, synchronously, back to back
# for each, will run in subprocess and show output live using reader/writers
# once finished, with exit code 0, will write to comletion hash

import argparse
from pathlib import Path
import hashlib
import subprocess
import shlex


def check_valid_workspace(path):
    manifest_dir = path / 'manifest'
    manifest_file = manifest_dir / 'manifest.txt'
    if not path.exists():
        raise Exception("Workspace folder does not exist")
    if not (manifest_dir.exists() and manifest_dir.is_dir()):
        raise Exception("Workspace needs to have a 'manifest' subdirectory")
    if not manifest_file.exists():
        raise Exception("Workspace needs to have a 'manifest/manifest.txt' file")


def get_job_hash(job_str):
    job_str = str(job_str.split()).encode('utf-8')
    return hashlib.sha1(job_str).hexdigest()


def parse_manifest(manifest_path):
    job_hashes = {}
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            job_hash = get_job_hash(line)
            job_hashes[job_hash] = line
    # for k, v in job_hashes.items():
    #     print(f'{k}: {v}')
    return job_hashes


def check_completions(completions_path, job_hashes):
    finished_jobs = []
    if completions_path.exists():
        with open(completions_path, 'r') as f:
            finished_jobs = f.read().splitlines()
    print('Status of jobs in this workspace ([<completed?>] <job command>):\n')
    need_to_run = {}
    for job_hash, cmd in job_hashes.items():
        if job_hash in finished_jobs:
            status = '[x]'
        else:
            status = '[ ]'
            need_to_run[job_hash] = cmd
        print(f'{status} {cmd[:72]}...')
    return need_to_run


def run_job_helper(cmd):
    cmd = 'python alignment_experiment.py ' + cmd
    print(cmd)
    completion = subprocess.run(cmd.split())
    return completion.returncode


def prefix_workspace_folder(cmd, workspace_folder):
    cmd = cmd.split()
    output_path = str(workspace_folder / cmd[1])
    cmd[1] = output_path
    return ' '.join(cmd)


def run_job(job_hash, cmd, workspace_folder):
    cmd = prefix_workspace_folder(cmd, workspace_folder)
    return_code = run_job_helper(cmd)
    if return_code == 0:
        with open(workspace_folder / 'manifest' / 'completions.txt', 'a+') as f:
            f.write(f'{job_hash}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('experiment-runner', description='Run list of specified experiments automatically.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder', help='Working space folder which contains manifest folder and file, and is where results are outputted.', type=Path)
    args = parser.parse_args()
    check_valid_workspace(args.folder)

    job_hashes = parse_manifest(args.folder / 'manifest' / 'manifest.txt')
    job_hashes = check_completions(args.folder / 'manifest' / 'completions.txt', job_hashes)

    for job_hash, cmd in job_hashes.items():
        run_job(job_hash, cmd, args.folder)