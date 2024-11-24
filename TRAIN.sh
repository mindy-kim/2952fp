#!/bin/bash

# job name
#SBATCH -J GAUSSIAN_GROUPING

#
#SBATCH --account=ssrinath-gcondo

# partition
#SBATCH --partition=ssrinath-gcondo --gres=gpu:1 --gres-flags=enforce-binding

# ensures all allocated cores are on the same node
#SBATCH -N 1

# cpu cores
#SBATCH --ntasks-per-node=4

# memory per node
#SBATCH --mem=32G

# runtime
#SBATCH -t 24:00:00

# output
#SBATCH -o sbatch_out.out

# error
#SBATCH -e sbatch_err.err

# email notifiaction
# SBATCH --mail-type=ALL

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate gaussian_splatting
module load cuda/11.8.0-lpttyok
module load opencv
module load ffmpeg
pip install --upgrade submodules/simple-knn
pip install --upgrade submodules/diff-gaussian-rasterization
python3 train.py -s data/brics-studio-pc
