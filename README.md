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

## Slurm basics

In the following, the procedure for launch works with slurms is detailed step by step. The next image provide a simple overview of the hardware machine available.

![hardware_basics](https://github.com/smorenospace/SlurmTorchHarmony/assets/169695104/a0ed319e-9c25-42bc-b1ac-f2eaaea186c7)=250x250

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

Here, the slurm variable --ntasks-per-node launch a specific number of process in each node. Therefore, there is no need to launch multiprocessing process by the interface of pytorch.

2. Launch processes by torchrun.

        sbatch launch_slurm_with_torchrun.sh

Here, torchrun automatically launch the number of processes by the specific features. 

3. Launch processes by torch multiprocessing.

       sbatch launch_slurm_with_torchmultiprocessing.sh

Here, the developer takes control of each of the required processes in the python code.
