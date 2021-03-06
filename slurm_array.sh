#!/bin/bash

COMMANDS_FILE=$1
echo ${COMMANDS_FILE}
mapfile -t job_commands < $COMMANDS_FILE

. /home/aalavi/anaconda2/etc/profile.d/conda.sh
echo "conda sourced"
conda activate point-align-gpu
which python
eval ${job_commands[$SLURM_ARRAY_TASK_ID]}
