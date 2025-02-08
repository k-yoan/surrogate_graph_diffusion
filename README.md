# Surrogate Models for Diffusion on Graphs: A High-Dimensional Polynomial Approach

This repository contains the code and data for the numerical experiments conducted as part of the paper "Surrogate Models for Diffusion on Graphs via Sparse Polynomial Approximation", which is based on Kylian Ajavon's Master's thesis research at Concordia University. The paper is co-authored by Kylian Ajavon and Dr. Simone Brugiapaglia, with Dr. Giuseppe Alessio D’Inverno as the lead author.

## Overview

This repository includes the implementation of the numerical experiments described in Section 4 of the paper, where we proposed a novel approach for approximating the state of a graph via high-dimensional polynomials. The experiments involve reproducing the following convergence plots:
- average Root Mean Square Error (RMSE) vs. number of sample points on Stochastic Block Models (SBMs),
- average RMSE vs. number of sample points on a Twitter graph dataset.

## Structure

- `src_sbm/`: Contains the source code for the numerical methods and experiment setups on Stochastic Block Models.
- `src_twitter/`: Contains the source code for the numerical methods and experiment setups on a Twitter dataset.
- `data/`: Data files used in the Twitter experiments.
- `results/`: Output data from the numerical experiments.

## Installation

### Prerequisites

- Python 3.

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

To run the experiments, follow these steps:
1. to be completed...
2. ...

## Results

Section to showcase some results from the paper...?

