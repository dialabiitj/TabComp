#!/bin/bash
# Job name:
#SBATCH --job-name=test
# Partition:
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
## Processors per task:
#SBATCH --cpus-per-task=2
#
#SBATCH --gres=gpu:2 ##Define number of GPUs

python test.py --dataset_name_or_path "./dataset/docvqa" --pretrained_model_name_or_path "/iitjhome/pratiwi1/donut/result/train_docvqa/20240629_021705" --save_path ./result/Mono_base.json
