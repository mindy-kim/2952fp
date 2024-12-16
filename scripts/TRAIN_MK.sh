#!/bin/bash

# job name
#SBATCH -J work

# partition
#SBATCH --partition=gpu --gres=gpu:1 --gres-flags=enforce-binding

# ensures all allocated cores are on the same node
#SBATCH -N 1

# cpu cores
#SBATCH --ntasks-per-node=4

# memory per node
#SBATCH --mem=128G

# runtime
#SBATCH -t 24:00:00

# output
#SBATCH -o sbatch_out.out

# error
#SBATCH -e sbatch_err.err

# email notifiaction
# SBATCH --mail-type=ALL

module load anaconda
module load cuda/11.8.0-lpttyok
source activate cs2952
# pip list
# pip install omegaconf
# pip install pytorch_lightning
python main.py -c configs/hijack1_full_lsamlp.yaml
python main.py -c configs/hijack2_full_lsamlp.yaml
python main.py -c configs/hijack4_full_lsamlp.yaml
python main.py -c configs/hijack8_full_lsamlp.yaml

python main.py -c configs/hijack1_5_full_lsamlp.yaml
python main.py -c configs/hijack1_10_full_lsamlp.yaml
python main.py -c configs/hijack1_20_full_lsamlp.yaml