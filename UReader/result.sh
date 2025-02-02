#!/bin/bash
# Job name:
#SBATCH --job-name=ureader
# Partition:
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
## Processors per task:
#SBATCH --cpus-per-task=2
#
#SBATCH --gres=gpu:2 ##Define number of GPUs

python -m pipeline.eval_utils.run_evaluation