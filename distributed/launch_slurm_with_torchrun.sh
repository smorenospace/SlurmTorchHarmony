#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --partition=volta
#SBATCH --time=600

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=1gb


### get the first node name as master address - customized for vgg slurm
### e.g. master(hermes,inf-004-gpu-4) == gnodee2

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12340
export CUDA_VISIBLE_DEVICES=0,1
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed (conda...)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate HSIap

### the command to run (same parameters than #SBATCH upper config)
srun torchrun \
--nnodes 2 \
--nproc_per_node 2 \
--rdzv_backend c10d \
main_slurm_to_torch.py --lr 1e-3 --epochs 200
