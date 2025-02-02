#!/bin/bash
# Job name:
#SBATCH --job-name=test_base_pretrained
# Partition:
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
## Processors per task:
#SBATCH --cpus-per-task=2
#
#SBATCH --gres=gpu:2 ##Define number of GPUs

python train.py --config config/train_docvqa.yaml