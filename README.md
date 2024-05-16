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

provides a comprehensive codebase for NLP distributed training using Slurm. This repository includes various alternatives for executing distributed training, encompassing all possibilities with Slurm and Torch code integration.

## Installation
        wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
        ./Anaconda3-2023.09-0-Linux-x86_64.sh (during installation set PREFIX=/home/<USER>/anaconda3)
        git clone https://github.com/smorenospace/SlurmTorchHarmony.git
        cd SlurmTorchHarmony/
        conda install requirements.yml or pip install requirements.txt

## Slurm basics

In the following, the procedure for launch works with slurms is detailed step by step. The next image provide a simple overview of the hardware machine available and how slurm asign the resources using this code example:

        sbatch 
        –node=localhost
        –cores-per-socket=5
        –mem=16000
        –mem-per-cpu=3200
        –exclusive
        –gpus-per-node=2 ./example.sh


<h1 align="center">
    <img src="https://github.com/smorenospace/SlurmTorchHarmony/assets/169695104/a0ed319e-9c25-42bc-b1ac-f2eaaea186c7" alt="hardware_basics" width="360" height="220">
    <img src="https://github.com/smorenospace/SlurmTorchHarmony/assets/169695104/9308c7e0-35fc-4ffe-b353-fce4a5dc566f" alt="slurm_launch" width="360" height="220">
</h1>


Next, 

## Tree Structure and Contents

> The directory structure provided in this repository is as follows:

    .
    ├── distributed
    │   ├── file11.ext
    │   └── file12.ext
    ├── local
    │   ├── file21.ext
    │   ├── file22.ext
    │   └── file23.ext
    ├── aux
    └── README.md

Below is a detailed point-by-point description of each code component.

### Distributed

1. Launch processes by slurm library.

        sbatch launch_slurm_to_torch.sh

Here, the slurm variable --ntasks-per-node launch a specific number of process in each node. Therefore, there is no need to launch multiprocessing process by the interface of pytorch. The following example details this operation with the respective image:

        sbatch –node=localhost
        –cores-per-socket=5
        –mem=16000
        –mem-per-cpu=3200
        –exclusive
        –gpus-per-node=2
        –ntasks-per-node=2
        –gpus-per-task=1
        –cpus-per-task=5 ./example.sh

<h1 align="left">
    <img src="https://github.com/smorenospace/SlurmTorchHarmony/assets/169695104/d176fedb-fcc9-4454-afdf-48462f0461c1" alt="slurm_to_torch" width="380" height="240">
</h1>

2. Launch processes by torchrun.

        sbatch launch_slurm_with_torchrun.sh

Here, torchrun automatically launch the number of processes by the specific features. 

3. Launch processes by torch multiprocessing.

       sbatch launch_slurm_with_torchmultiprocessing.sh

Here, the developer takes control of each of the required processes in the python code.
