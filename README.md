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

## Tree Structure and Contents

The directory structure provided in this repository is as follows:

### A typical top-level directory layout

    .
    ├── build                   # Compiled files (alternatively `dist`)
    ├── docs                    # Documentation files (alternatively `doc`)
    ├── src                     # Source files (alternatively `lib` or `app`)
    ├── test                    # Automated tests (alternatively `spec` or `tests`)
    ├── tools                   # Tools and utilities
    ├── LICENSE
    └── README.md

Below is a detailed point-by-point description of each code component:
