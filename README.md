# Non-Stationary Inventory Control With Lead Times

**Manuscript:** Non-Stationary Inventory Control With Lead Times

**Authors:** Nele H. Amiri, Sean R. Sinclair, Maximiliano Udenio

---

## 1. Overview

This repository contains the code required to reproduce the numerical experiments in the paper *Non-Stationary Inventory Control With Lead Times*. The codebase implements the NSIC-BL, NSIC-LS, and NSIC-LSL learning algorithms for inventory control problems characterized by switching demand, accommodating both backlogging and lost-sales models with non-negative lead times. The repository provides code for the evaluation of the algorithm's performance on simulated data as reported in the manuscript.
The code is written in an object-oriented manner so that it can be easily supplemented with additional (benchmark) algorithms or classes of problem instances.

---

## 2. Computational requirements

All code is written in **Python (>= 3.12)**.

Key dependencies include:
- `numpy` for efficient numerical computations and matrix operations
- `scipy` for probabilistic modeling and simulation
- `pyyaml` for providing input parameters to specify simulation instances
- `optuna` (optional) for optimization of hyperparameters/constants

The code implements multiprocessing to parallelize simulation runs across available CPU cores.

---

## 3. Program structure 

### File overview
- `main.py`: runs simulation or performs hyperparameter optimization
- `input\configSimulation.yaml`: specifies input parameters
- `algorithms\algoClass.py` and `algorithms\algorithm.py`: define algorithm class and inherited class NSIC, respectively.
- `environments\envClass.py`: defines class `Environment` which represents the environment (demand sampling, update of inventory state, compuation of costs, change in demand distribution etc.)
- `helpers\helpers.py`: validation of input parameters, initialization of environment, computation of best fixed base-stock policy in hindsight (for computation of static regret)
- `helpers\distributionClass.py`: defines a class for each supported demand distribution family (Normal, Uniform, Poisson, Exponential)

### Inputs

All datasets are generated using the scripts provided in this repository. No external data sources are required. Specifically, the simulation setup is specified in the user input file `configSimulation.yaml` with the demand distributions defined in `distributionClass.py` and initialized in `helpers.py`.

All random number generation uses fixed seeds where appropriate to ensure reproducibility.

### Output
Each execution of the algorithm through `main.py` produces CSV files with the performance metrics outlined below, saved in the `logs` folder.

---

## 5. Variable dictionaries

Below we describe the key variables used across scripts. Where practical, the variable names from the manuscript were used.

### Common variables
- `T`: time horizon
- `L`: lead time
- `S`: number of stationary time intervals over horizon T (number of change points minus one)
- `b`: underage (stockout/backlog) cost
- `h`: holding cost
- `model`: inventory model in which underage demand is backlogged (`backlog`) or lost (`lost_sales`)
- `U`: upper bound on optimal base-stock level
- `bslevels`: array of discrete set of base-stock levels
- `K`: number of discrete base-stock levels with `bslevels[0] = 0.0` and `bslevels[K-1] = U`
- `d_distribution`: demand distribution 

### Metrics
- `dynamicRegret`: dynamic regret relative to sequence of optimal base-stock levels given by the random sequenece of demand distributions
- `staticRegret`: static regret relative to best fixed base-stock level in hindsight
- `trueCost`: realized costs incurred by sequence of base-stock levels selected
- `expectedCost`: extected costs of sequence of base-stock levels selected
- `relatveRegret`: relative regret of sequence of base-stock levels selected


