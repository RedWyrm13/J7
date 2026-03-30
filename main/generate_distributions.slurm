#!/bin/bash
#SBATCH --job-name=J7_Data_mb
#SBATCH --output=multi_basis.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amacieju@odu.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12

module load python3/2025.1-py312

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Create output directories for all three measurement modes
mkdir -p ../data/quantum
mkdir -p ../data/quantum_mb
mkdir -p ../data/quantum_shadows

crun -p /home/amaci003/envs/J7_sampling python -u generate_distributions.py circuit_configuration.yaml
