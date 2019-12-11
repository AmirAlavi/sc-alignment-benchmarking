#!/bin/bash

CONTAINER=/containers/images/u1604-cuda-9.1_pytorch_tensorflow.img
SCRIPT=$1

echo "hi"
echo ${SCRIPT}
pwd
singularity exec --nv ${CONTAINER} /bin/bash ${SCRIPT} ${@:2}
