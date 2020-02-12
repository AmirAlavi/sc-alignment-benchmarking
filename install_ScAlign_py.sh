#/bin/bash
cd ..
. "/home/amir/miniconda3/etc/profile.d/conda.sh"
git clone https://github.com/AmirAlavi/scAlign-py.git
conda activate point-align
cd scAlign-py/python
pip install -e .