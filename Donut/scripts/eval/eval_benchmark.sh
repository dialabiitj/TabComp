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

export PYTHONPATH=`pwd`
python -m torch.distributed.launch --use_env \
    --nproc_per_node ${NPROC_PER_NODE:-2} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    pipeline/evaluation.py \
    --hf_model /iitjhome/pratiwi1/UReader/checkpoints/mplug-owl-llama-7b