<h1 align="center">
    <a href="https://slurm.schedmd.com/quickstart.html">
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Slurm_logo.svg" width="100" height="90">
    </a>
    <a href="https://pytorch.org/tutorials/beginner/dist_overview.html">
    <img src="https://heidloff.net/assets/img/2023/09/python-pytorch.png" width="160" height="90">
    </a>
</h1>



<p align="center">
  <i align="center">🚀 Code examples and best practices for distributed [slurm+torch] for NLP 🚀</i>
</p>

## Introduction

This package provides a comprehensive codebase for NLP distributed training using Slurm. The repository includes various alternatives for executing distributed training, encompassing all possibilities with Slurm and Torch code integration.

## Installation
        wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
        ./Anaconda3-2024.10-1-Linux-x86_64.sh --prefix /home/$USER/data/anaconda3 (during installation set PREFIX=/home/$USER/data/anaconda3)
        git clone https://github.com/smorenospace/SlurmTorchHarmony.git
        cd SlurmTorchHarmony/
        conda install requirements.yml

## Slurm basics

In the following, the procedure for launch works with slurms is detailed step by step. $${\textcolor{blue}{NEVER \space LAUNCH \space CODE \space WITH \space PYTHON, \space ALWAYS \space USE \space SLURM \space THROUGH \space THE \space FOLLOWING \space COMMANDS}}$$. The next image provide a simple overview of the hardware machine available (yellow=idle, red=alloc, black=restricted) and how slurm asign the resources using this code example:

        sbatch 
        --nodes=1
        --p=A30
        --node=hermes
        --gres=gpu:A30:2
        --gpus-per-node=2 ./example.sh

Alternatively, you can use srun for python files. Check https://slurm.schedmd.com/srun.html for more information.

<h1 align="center">
    <img src="https://github.com/smorenospace/SlurmTorchHarmony/assets/169695104/a0ed319e-9c25-42bc-b1ac-f2eaaea186c7" alt="hardware_basics" width="360" height="220">
    <img src="https://github.com/smorenospace/SlurmTorchHarmony/assets/169695104/9308c7e0-35fc-4ffe-b353-fce4a5dc566f" alt="slurm_launch" width="360" height="220">
</h1>

The information about the machines can be stated by the "sinfo" command. The output is the following:

| PARTITION | AVAIL | TIMELIMIT | NODES | STATE | NODELIST
| --- | --- | --- | --- | --- | --- |
| A5000 | up | infinite | 2 | idle or mix or alloc | inf-004-gpu-4, inf-004-gpu-3
| A30 | up | infinite | 1 | idle or mix or alloc | hermes

To check the state of your job, the queue can be display by using the "squeue" command. The output is the following:

| JOBID | PARTITION | NAME | USER | STATE | TIME | NODES | NODELIST (REASON in case not running) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 124 | A5000 | example.sh | smoreno | RUNNING | 04:23 | 2 | inf-004-gpu-4, inf-004-gpu-3 |
| 125 | A5000 | example2.sh | jrodri | PENDING | 0:00 | 2 | RESOURCES (waiting resources) |

## Tree Structure and Contents

> The directory structure provided in this repository is as follows:

    .
    ├── distributed
    │   ├── launch_slurm_to_torch.sh (slurm launch file 1st alternative)
    │   ├── launch_slurm_with_torchrun.sh (slurm launch file 2nd alternative)
    │   ├── launch_slurm_with_accelerate.sh (slurm launch file 3rd alternative)
    │   ├── loader.py (dataloader)
    │   ├── trainer.py (train and test functions)
    │   ├── utils.py (auxilary functions)
    │   ├── chkpt (should be created on run)
    │   │   ├──  version-id
    │   │   │   ├── acc (output accuracy logs)
    │   │   │   ├── loss (output loss logs)
    │   │   │   ├── lr (learning rate logs)
    │   │   │   ├── events.out.tfevents (tensorboard)
    │   │   │   ├── experiment.log (log)
    ├── convergence_proof
    └── README.md

Below is a detailed point-by-point description of each code component based on 4 launch alternatives.

### Launch

1. Launch processes by slurm library.

        sbatch --nodes=1 --gres=gpu:A5000:4 --partition=A5000 --time=16:00:00 --tasks-per-node=4 ./launch_slurm_to_torch.sh

Here, the slurm variable --ntasks-per-node launch a specific number of process in each node. Therefore, there is no need to launch multiprocessing process by the interface of pytorch. The following example details this operation with the respective image:

        sbatch –node=machine1 (--nodes=1 in case you are not interest in a specific machine)
        –cores-per-socket=5
        –mem=16000
        –mem-per-cpu=3200
        –exclusive
        –gpus-per-node=2
        –ntasks-per-node=2
        –gpus-per-task=1
        –cpus-per-task=5 ./example.sh

<h1 align="left">
    <img src="https://github.com/smorenospace/SlurmTorchHarmony/assets/169695104/d176fedb-fcc9-4454-afdf-48462f0461c1" alt="slurm_to_torch" width="360" height="220">
</h1>

2. Launch processes by torchrun.

        sbatch --nodes=1 --gres=gpu:A5000:4 --partition=A5000 --time=16:00:00 ./launch_slurm_with_torchrun.sh

Before the torchrun statement, the execution of the previous command generates the following slurm configuration (left next image):

<h1 align="center">
        <img src="https://github.com/smorenospace/SlurmTorchHarmony/assets/169695104/a08a8b8c-571b-4b91-b8a5-5071a0bf6258" alt="torchrun" width="340" height="240">
        <img src="https://github.com/smorenospace/SlurmTorchHarmony/assets/169695104/d176fedb-fcc9-4454-afdf-48462f0461c1" alt="slurm_to_torch" width="360" height="240">
</h1>

Then, torchrun automatically launch the number of processes by the specific features (right previous image). Therefore, we only launch 1 slurm task. The example code contained in launch_slurm_with_torchrun.sh is similar to:

        #!/bin/bash
        CUDA_VISIBLE_DEVICES=0,1
        srun torchrun --standalone --nnodes=1 --nproc-per-node=2
           YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

where, YOUR_TRAINING_SCRIPT.py must contain at some point the fix on processes to gpu based on local ranks.

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank) #0,1 cuda devices


3. Launch processes with accelerate (for this one, an example of finetuning RACE is provided using 2 nodes and 8 GPUs on A5000 partition)
   
       sbatch --nodes=2 --gres=gpu:A5000:4 --partition=A5000 --time=16:00:00 ./launch_slurm_with_accelerate.sh

4. (MORE OPTIONS NOT INCLUDED IN THE REPO)
   Launch processes by torch multiprocessing.

       sbatch launch_slurm_with_torchmultiprocessing.sh

Here, the developer takes control of each of the required processes in the python code. Similar to the previous point 2, we only launch 1 slurm task. The example code is similar to:

        #!/bin/bash
        CUDA_VISIBLE_DEVICES=0,1
        python YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

where, after spawn the specific number of processes, YOUR_TRAINING_SCRIPT.py must contain at some point the fix on processes to gpu based on local ranks. The workaround is the same than the previous two Figures from point 2.

        mp.spawn(nprocs=2)
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank) #0,1 cuda devices


### Core idea

The main concern is to correctly create a **distributed environment within torch and asign the respective local ranks from slurm to the respective GPUs**. Therefore, these two steps must be ensured for points 1 and 3. On the other hand, point 2 only require gpu assign since the distributed environment is already launch. To set the gpus use torch.cuda.set_device. The Step [0] of the code prints this information for 2 nodes with 2 gpus (ranks 0 and 1).

        [Step 0] Check GPUs and process assigment
        _____________________________________________
        This is process number  0  set to GPU device number (local rank: 0 == local gpu: 0 )
        _____________________________________________
        [Step 0] Check GPUs and process assigment
        _____________________________________________
        This is process number  3  set to GPU device number (local rank: 1 == local gpu: 1 )
        _____________________________________________
        [Step 0] Check GPUs and process assigment
        _____________________________________________
        This is process number  1  set to GPU device number (local rank: 1 == local gpu: 1 )
        _____________________________________________
        [Step 0] Check GPUs and process assigment
        _____________________________________________
        This is process number  2  set to GPU device number (local rank: 0 == local gpu: 0 )
        _____________________________________________


