# HyperSHAP

This repository contains the implementation of the HyperSHAP hyperparameter importance games.
The code is integrated with the ``shapiq`` Python package, mirrored from the original [repository](https://github.com/mmschlk/shapiq).
Furthermore, we provide code for all plots appearing in the main paper as well as in the appendix. This also includes
implementations for the downstream task of conducting HPO after determining HPI. Together with the implementation, we
provide pre-computed games in the folders ``game_storage/`` and ``hpo_storage/``.

## What is not included?
We do not include the data for YAHPOGym, PD1 nor JAHS-Bench-201.
The data for YAHPOGym can be downloaded from the following URLs:
- https://github.com/slds-lmu/yahpo_data.

The data packages for pd1 and JAHS-Bench-201 should be automatically downloaded with installing the corresponding environments.
To ensure compatibility, the folder structure should look like this
````
/<hypershap-root-dir>
    data/
        jahs/
            assembled_surrogates/
                ...
        pd1/
            surrogates/
                ...
        yahpodata/
            benchmark_suites/
            fcnet/
            global_statistics/
            iaml_glmnet/
            ...
````

## Environment Setup
The requirements for HyperSHAP are served with the requirements.txt file. However, for different benchmarks, due to dependency
conflicts different environments are needed. Please refer to the installation bash scripts to setup environments accordingly.
However, if you would like to also run the downstream tasks, you need to install [SMAC3](https://github.com/automl/SMAC3) first, following their installation guidelines.

# Overview of the Code Supplement
In this section, we give an overview of the important parts of the code submission for reproducing the experiments and plots of the paper.

## Benchmarks
We wrap benchmarks used for evaluating HyperSHAP in the following folder:
````
/<hypershap-root-dir>
    hypershap/
        base/
            benchmark/
                __init__.py
                abstract_benchmark.py
                ...
````
If you would like to use HyperSHAP for your specific hyperparameter optimization problem, you will need to add another
benchmark that implements the base class ``AbstractBenchmark``.

## Implementation of Games
All the games presented in the paper are located in 
````
/<hypershap-root-dir>
    hypershap/
        base/
            games/
                __init__.py
                ablation.py
                abstract_hpi_game.py
                optimizer_bias.py
                tunability.py
````

If you want to execute games yourself, you can do so by running ``python -m hypershap.precompute.pre_compute_games.py``;
running downstream tasks works by running ``prec_compute_downstream_hpo.py``.

## Plotting
All plots presented in the paper can be reproduced by running the Python scripts that are located in the following folder:
````
/<hypershap-root-dir>
    hypershap/
        plots/
            ...
````