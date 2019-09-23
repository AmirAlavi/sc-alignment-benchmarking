#!/bin/bash

COMMANDS_FILE=$1

mapfile -t job_commands < $COMMANDS_FILE

eval ${job_commands[$SLURM_ARRAY_TASK_ID]}