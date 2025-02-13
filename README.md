# Surrogate Models for Diffusion on Graphs via Sparse Polynomials

This repository contains the code and data for the numerical experiments conducted as part of the paper "Surrogate Models for Diffusion on Graphs via Sparse Polynomials" (https://arxiv.org/abs/2502.06595), co-authored by Giuseppe Alessio D'Inverno, Kylian Ajavon and Simone Brugiapaglia.

In this paper, we explored diffusion processes on graphs and proposed a novel approach for approximating the state of a graph via sparse polynomial expansions. Our approach can help overcome the computational challenges of simulating diffusion processes on large-scale systems, and we demonstrate its accuracy in approximating these diffusion processes through the numerical experiments in this repository.

## Overview

This repository includes the implementation of the numerical experiments described in Section 4 of the paper. The experiments involve reproducing the following convergence plots:
- average Root Mean Square Error (RMSE) vs. number of sample points on Stochastic Block Models (SBMs),
- average RMSE vs. number of sample points on a Twitter graph dataset.

## Structure

- `results/`: Output data from the numerical experiments and data files used in the Twitter experiments.
- `src_sbm/`: Contains the source code for the numerical methods and experiment setups on Stochastic Block Models.
- `src_twitter/`: Contains the source code for the numerical methods and experiment setups on a Twitter dataset.

## Installation

### Prerequisites

- Python 3.7

### Clone the repository:
```bash
git clone https://github.com/k-yoan/surrogate_graph_diffusion
cd surrogate_graph_diffusion
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run reconfiguration file:
After installing the dependencies, you need to apply custom modifications to the `equadratures` library. This step is essential for our experiments to work as intended. To handle the modifications, we have created a Python script that automatically reconfigures the library. Run the following command to apply the changes:
```bash
python src_sbm/edit_equad.py
```

## Running the Experiments

We have created Python scripts that allow you to run the experiments and generate the figures. You can use the argument parser in each script to easily adjust the hyperparameters directly from the terminal. Here are some arguments you can tune:
- `--nb_communities`: Set the number of communities in the Stochastic Block Model.
- `--nodes_per_comm`: Set the number of nodes in each community.
- `--basis`: Select which multi-index set to use as a basis.
- `--order`: Set the order of the multi-index set.
- `--n_trial`: Set the number of rounds of computation for each experiment.

For example:
```bash
cd src_sbm
python convergence_by_method.py --nb_communities 3 --basis 'hyperbolic-cross' --order 15
```
Or to use the default values, simply run:
```bash
cd src_sbm
python convergence_by_method.py
```

Here are the available Python scripts to run:
- `convergence_by_method.py`
- `convergence_by_graph_size.py`

Make sure you are running the Python scripts from the `src_sbm` directory.


