#!/bin/bash

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=12340                     # any free TCP port

GPUS_PER_NODE=4                              # local GPUs *on each node*
NUM_PROCS=8

echo "[batch] MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "[batch] NODES=$SLURM_NNODES  GPUS_PER_NODE=$GPUS_PER_NODE"

# ---------------------------------------------------------------------
# 1. spawn one Accelerate *front-end* per node
#    Slurm sets $SLURM_PROCID (0…nodes-1)
# ---------------------------------------------------------------------
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES bash -c '
  echo "[front-end] global_rank=$SLURM_PROCID  local_rank=$SLURM_LOCALID  host=$(hostname)"

  accelerate launch \
    --num_processes '"$NUM_PROCS"' \
    --num_machines  '"$SLURM_NNODES"' \
    --machine_rank  $SLURM_PROCID \
    --main_process_ip   '"$MASTER_ADDR"' \
    --main_process_port '"$MASTER_PORT"' \
    finetuning.py
'
