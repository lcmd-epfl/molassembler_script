#!/bin/bash
conda env create --file masb_kit.yaml 
conda init bash
if [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
    source ~/miniconda3/etc/profile.d/conda.sh
else 
    echo "Could not find the miniconda/anaconda path. Edit installer.sh."
    exit
fi
conda activate masb_kit
pip install scine_utilities-3.0.0-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install scine_database-1.0.0-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install scine_molassembler-1.1.0-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

