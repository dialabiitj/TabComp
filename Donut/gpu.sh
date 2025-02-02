#!/bin/bash
# Job name:
#SBATCH --job-name=test
# Partition:
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
## Processors per task:
#SBATCH --cpus-per-task=2
#
#SBATCH --gres=gpu:2 ##Define number of GPUs
nvidia-smi