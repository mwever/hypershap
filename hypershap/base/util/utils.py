"""This module contains utility functions for running the experiments."""

import os
from typing import Optional

import numpy as np
import shapiq

from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark
from hypershap.base.benchmark.jahs import JAHSBenchmark
from hypershap.base.benchmark.pd1 import PD1Benchmark
from hypershap.base.benchmark.yahpogym import YahpoGymBenchmark
from hypershap.base.games.ablation import AblationHPIGame
from hypershap.base.games.optimizer_bias import DataSpecificOptimizerBiasGame
from hypershap.base.games.tunability import (
    DataSpecificTunabilityHPIGame,
    TunabilityHPIGame,
)
from hypershap.base.optimizer.local_optimizer import LocalOptimizer
from hypershap.base.optimizer.random_optimizer import SubspaceRandomOptimizer
from hypershap.base.util.constants import (
    ABLATION_GAME,
    DS_TUNABILITY_GAME,
    OB_GAME_SET_A_SUFFIX,
    OB_GAME_SET_B_SUFFIX,
    OPTIMIZER_BIAS_GAME,
    TUNABILITY_GAME,
)

GAME_STORAGE_DIR = "game_storage"
os.makedirs(GAME_STORAGE_DIR, exist_ok=True)


def _find_optimal_configuration(
    hpo_problem: HyperparameterOptimizationBenchmark,
    random_state: Optional[int] = 42,
    n_configs: int = 10_000,
) -> tuple[dict, float]:
    """Find the optimal configuration for the given benchmark.

    Args:
        hpo_problem (HyperparameterOptimizationBenchmark): the HPO benchmark problem.
        random_state: The random state. Default is `42`.
        n_configs: The number of configurations to sample. Default is `1000`.

    Returns:
        A tuple containing the optimal configuration and its value.
    """

    opt_config, opt_config_value = None, None
    configurations = hpo_problem.sample_configurations(n_configs, random_state=random_state)
    for config in configurations:
        config_value = hpo_problem.evaluate(config)
        if opt_config_value is None or config_value > opt_config_value:
            opt_config, opt_config_value = config, config_value
    return opt_config, opt_config_value


def _get_game_name(
    game_type: str,
    hpo_problem: HyperparameterOptimizationBenchmark,
    **kwargs,
) -> str:
    """Get the name of the game.

    Args:
        game_type: The game to get the name for.
        benchmark_name: The name of the benchmark.
        **kwargs: Additional keyword arguments to be passed to the game name.

    Returns:
        The name of the game.
    """
    game_name = game_type
    game_name += f"_{hpo_problem.benchmark_lib}_{hpo_problem.scenario}_{hpo_problem.dataset}_{hpo_problem.metric}"
    for key, value in kwargs.items():
        game_name += f"_{key}={value}"
    return game_name


def setup_game(
    game_type: str,
    benchmark: str,
    scenario: str,
    metric: str = "val_accuracy",
    instance_index: Optional[int | str] = None,
    random_state: Optional[int] = 42,
    n_configs: int = 10_000,
    pre_compute: bool = False,
    verbose: bool = False,
    normalize_loaded: bool = True,
    only_load: bool = False,
    n_instances_universal: Optional[int] = None,
) -> tuple[shapiq.Game, str, list[str], HyperparameterOptimizationBenchmark]:
    """Sets up the hyperparameter importance game.

    Args:
        game_type: The type of game to set up.
        benchmark_name: The name of the benchmark.
        metric: The metric to optimize. Default is `"acc"`.
        instance_index: The instance to use. If None, the first instance is used. Default is `None`.
        random_state: The random state. Default is `42`.
        n_configs: The number of configurations to sample. Default is `1000`.
        pre_compute: Whether to pre-compute and store the game values. Default is `False`.
        verbose: Whether to print additional information. Default is `False`.
        normalize_loaded: Whether to normalize the loaded game values. Default is `True`.
        only_load: Whether to only load games from disk (`True`) or also to initialize them from
            scratch (`False`). Default is `False`.
        n_instances_universal: The number of instances to use for the universal game. Default is
            `None`.

    Returns:
        The hyperparameter importance game, the name of the game, and the player names.

    Raises:
        ValueError: If the game is not found and `only_load` is set to `True`.
    """

    # compile hpo benchmark problem from provided arguments
    if benchmark == "yahpogym":
        hpo_problem = YahpoGymBenchmark(
            scenario_name=scenario, metric=metric, instance_idx=instance_index
        )
    elif benchmark == "yahpogym-sense":
        hpo_problem = YahpoGymBenchmark(
            scenario_name=scenario, metric=metric, instance_idx=instance_index, sensitivity=True
        )
    elif benchmark == "pd1":
        hpo_problem = PD1Benchmark(scenario_name=scenario)
    elif benchmark == "jahs":
        hpo_problem = JAHSBenchmark(dataset=instance_index)
    else:
        ValueError("Invalid benchmark name.")

    # get game_name
    game_name = _get_game_name(
        game_type, hpo_problem, n_configs=n_configs, random_state=random_state
    )

    # check if the game is already stored
    game_path = os.path.join(GAME_STORAGE_DIR, f"{game_name}.npz")
    name_file = os.path.join(
        GAME_STORAGE_DIR,
        f"{hpo_problem.benchmark_lib}_{hpo_problem.scenario}_{hpo_problem.dataset}.names",
    )

    if os.path.exists(game_path) and os.path.exists(name_file):
        game = shapiq.Game(path_to_values=game_path, verbose=verbose, normalize=normalize_loaded)
        player_names = open(name_file).read().splitlines()
        print(f"Loaded game from {game_path}.")
        return game, game_name, player_names, hpo_problem

    if only_load:
        raise ValueError(
            f"Game {game_name} not found. Check the parameters if the game is already stored.\n"
            f"Parmaters: game_type={game_type}, benchmark_name={benchmark}, metric={metric},"
            f"instance_index={instance_index}, n_configs={n_configs}, random_state={random_state}"
        )

    shared_game_config = {"n_configs": n_configs, "random_state": random_state, "verbose": verbose}

    # set up the game from parameters
    if game_type == TUNABILITY_GAME:
        game = TunabilityHPIGame(hpo_problem, **shared_game_config)
    elif game_type == DS_TUNABILITY_GAME:
        game = DataSpecificTunabilityHPIGame(hpo_problem, instance_index, **shared_game_config)
    elif game_type == ABLATION_GAME:
        optimal_cfg, _ = _find_optimal_configuration(hpo_problem, random_state, n_configs)
        game = AblationHPIGame(hpo_problem, instance_index, optimal_cfg, hpo_problem.get_default_config(), random_state, verbose)
    elif game_type == OPTIMIZER_BIAS_GAME:
        optimizer = LocalOptimizer(random_state=random_state, verbose=verbose)
        random_search = SubspaceRandomOptimizer()
        game = DataSpecificOptimizerBiasGame(
            hpo_problem,
            instance=instance_index,
            ensemble=[optimizer],
            optimizer=optimizer,
            random_state=random_state,
            verbose=verbose,
        )
    elif game_type in [
        OPTIMIZER_BIAS_GAME + OB_GAME_SET_A_SUFFIX,
        OPTIMIZER_BIAS_GAME + OB_GAME_SET_B_SUFFIX,
    ]:
        param_set = list()
        if game_type == OPTIMIZER_BIAS_GAME + OB_GAME_SET_B_SUFFIX:
            for hyperparam in hpo_problem.get_opt_space().get_hyperparameters():
                if hyperparam.name not in ["OpenML_task_id", "epoch", "weight_decay"]:
                    param_set += [hyperparam.name]
        elif game_type == OPTIMIZER_BIAS_GAME + OB_GAME_SET_A_SUFFIX:
            param_set = ["learning_rate", "max_dropout", "max_units"]

        optimizer = SubspaceRandomOptimizer(param_set, random_state=random_state, verbose=verbose)
        game = DataSpecificOptimizerBiasGame(
            benchmark,
            metric=metric,
            optimizer=optimizer,
            n_configs=n_configs,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Invalid game type: {game_type}")

    print(game.n_players)

    # pre-compute and save the game values
    if pre_compute:
        game.precompute()
        game.save_values(game_path)

    # save the player names
    with open(name_file, "w") as f:
        f.write("\n".join(hpo_problem.get_list_of_tunable_hyperparameters()))
    player_names = hpo_problem.get_list_of_tunable_hyperparameters()

    return game, game_name, player_names, hpo_problem


def compute_avg_anytime_performance_lines(traces):
    max_length = None
    for t in traces:
        if max_length is None or len(t) > max_length:
            max_length = len(t)

    best_performance_profiles = list()
    for eval_trace in traces:
        best_value = None
        max_profile = list()
        for val in eval_trace:
            if best_value is None or val > best_value:
                best_value = val
            max_profile.append(best_value)

        while len(max_profile) < max_length:
            max_profile.append(best_value)

        best_performance_profiles.append(max_profile)

    best_perf_matrix = np.array(best_performance_profiles)
    avg_best_perf_list = list()
    std_best_perf_list = list()
    for i in range(best_perf_matrix.shape[1]):
        avg_best_perf_list.append(best_perf_matrix[:, i].mean())
        std_best_perf_list.append(best_perf_matrix[:, i].std())
    return np.array(avg_best_perf_list), np.array(std_best_perf_list)
