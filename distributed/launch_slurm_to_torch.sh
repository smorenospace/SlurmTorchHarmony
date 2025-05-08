#!/bin/bash


### get the first node name as master address - customized for vgg slurm
### e.g. master(hermes,inf-004-gpu-4) == gnodee2

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12340
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate HSIap

### the command to run
srun python main_slurm_to_torch.py --lr 1e-3 --epochs 200
