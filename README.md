# HyperSHAP

This repository contains the implementation of the HyperSHAP hyperparameter importance games.
The code is integrated with the ``shapiq`` Python package, mirrored from the original [repository](https://github.com/mmschlk/shapiq).
Furthermore, we provide code for all plots appearing in the main paper as well as in the appendix. This also includes
implementations for the downstream task of conducting HPO after determining HPI. Together with the implementation, we
provide pre-computed games in the folders ``game_storage/`` and ``hpo_storage/``.

## What is not included?
We do not include the backbone for YAHPO Gym which, can simply be downloaded from the following URL: https://github.com/slds-lmu/yahpo_data. To ensure compatibility,
the folder structure should look like this
````
/<hypershap-root-dir>
    yahpodata/
        benchmark_suites/
        fcnet/
        global_statistics/
        iaml_glmnet/
        ...
````

## Environment Setup
The requirements for HyperSHAP are served with the requirements.txt file, which includes everything necessary to pre compute games.
However, if you would like to also run the downstream tasks, you need to install [SMAC3](https://github.com/automl/SMAC3) first, following their installation guidelines.

# Overview of the Code Supplement
In this section, we give an overview of the important parts of the code submission for reproducing the experiments and plots of the paper.

## Implementation of Games
All the games presented in the paper are implemented in ``hpo_games.py``. The HPO methods for the downstream tasks are
implemented in ``downstream_hpo.py``.
If you want to execute games yourself, you can do so by running ``pre_compute.py``; running downstream tasks works by
running ``prec_compute_downstream_hpo.py``.

## Plotting
All plots presented in the paper can be reproduced by running the Python scripts that are prefixed with ``plot_``.
All in all, there are the following plotting scripts:
````
/<hypershap-root-dir>
    plot_downstream.py  # plotting anytime performances of downstream HPO
    plot_faithfulness_hpo_budget.py # plotting R2 faithfulness graphs over interaction orders
    plot_interactions.py  # plotting SI graphs independently of each other
    plot_moebius_interactions.py  # plotting violin plots over interaction orders
    plot_optimizer_bias.py  # plotting interaction graphs for the optimizer bias game
    plot_si_multiple.py  # plot SI graphs with joint normalization across multiple graphs 
````