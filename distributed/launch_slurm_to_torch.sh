#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --partition=volta
#SBATCH --time=60

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2gb


### get the first node name as master address - customized for vgg slurm
### e.g. master(hermes,inf-004-gpu-4) == gnodee2

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source ~/anaconda3/etc/profile.d/conda.sh
conda activate HSIap

### the command to run
srun python main_slurm_to_torch.py --lr 1e-3 --epochs 10
