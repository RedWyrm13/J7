#!/bin/bash
#SBATCH --job-name=J7_Data
#SBATCH --output=J7_Data.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amacieju@odu.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH -cpus-per-task=24

module load python3/2025.1-py312

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

crun -p /home/amaci003/envs/J7_sampling python -u generate_distributions.py circuit_configuration.yaml