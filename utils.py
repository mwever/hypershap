"""This module contains utility functions for running the experiments."""

import os
from typing import Optional

from tqdm import tqdm
from yahpo_gym import benchmark_set

import shapiq
from hpo_games import (
    GlobalHyperparameterImportanceGame,
    LocalHyperparameterImportanceGame,
    UniversalHyperparameterImportanceGame,
    UniversalLocalHyperparameterImportanceGame,
)

GAME_STORAGE_DIR = "game_storage"
os.makedirs(GAME_STORAGE_DIR, exist_ok=True)


__all__ = ["setup_game"]


def _find_optimal_configuration(
    benchmark: benchmark_set.BenchmarkSet,
    metric: str = "acc",
    random_state: Optional[int] = 42,
    n_configs: int = 1000,
) -> tuple[dict, float]:
    """Find the optimal configuration for the given benchmark.

    Args:
        benchmark: The benchmark set.
        metric: The metric to optimize. Default is `"acc"`.
        random_state: The random state. Default is `42`.
        n_configs: The number of configurations to sample. Default is `1000`.

    Returns:
        A tuple containing the optimal configuration and its value.
    """

    opt_config, opt_config_value = None, None
    search_space = benchmark.get_opt_space(drop_fidelity_params=False, seed=random_state)
    configurations = search_space.sample_configuration(n_configs)
    for config in configurations:
        config_dict = config.get_dictionary()
        config_value = benchmark.objective_function(config_dict)[0][metric]
        if opt_config_value is None or config_value > opt_config_value:
            opt_config, opt_config_value = config_dict, config_value
    return opt_config, opt_config_value


def _get_game_name(
    game_type: str,
    benchmark: str,
    **kwargs,
) -> str:
    """Get the name of the game.

    Args:
        game_type: The game to get the name for. Available games are:
            - universal: The `UniversalHyperparameterImportanceGame`.
            - global: The `GlobalHyperparameterImportanceGame`.
            - local: The `LocalHyperparameterImportanceGame`.
            - universal-local: The `UniversalLocalHyperparameterImportanceGame`.
        benchmark_name: The name of the benchmark.
        **kwargs: Additional keyword arguments to be passed to the game name.

    Returns:
        The name of the game.
    """
    game_name = game_type
    game_name += f"_{benchmark}"
    for key, value in kwargs.items():
        game_name += f"_{key}={value}"
    return game_name


def setup_game(
    game_type: str,
    benchmark_name: str,
    metric: str = "val_accuracy",
    instance_index: Optional[int] = None,
    random_state: Optional[int] = 42,
    n_configs: int = 1000,
    pre_compute: bool = False,
    verbose: bool = False,
    normalize_loaded: bool = True,
) -> tuple[shapiq.Game, str, list[str]]:
    """Sets up the hyperparameter importance game.

    Available game types are:
        - universal: The `UniversalHyperparameterImportanceGame`.
        - global: The `GlobalHyperparameterImportanceGame`.
        - local: The `LocalHyperparameterImportanceGame`.
        - universal-local: The `UniversalLocalHyperparameterImportanceGame`.

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

    Returns:
        The hyperparameter importance game, the name of the game, and the player names.
    """
    benchmark = benchmark_set.BenchmarkSet(benchmark_name)
    if instance_index is None:
        instance_index = benchmark.instances[0]
    else:
        instance_index = benchmark.instances[instance_index]

    # get game_name
    if game_type == "universal":
        game_name = _get_game_name(
            game_type, benchmark_name, metric=metric, n_configs=n_configs, random_state=random_state
        )
    else:
        game_name = _get_game_name(
            game_type,
            benchmark_name,
            metric=metric,
            instance=instance_index,
            n_configs=n_configs,
            random_state=random_state,
        )

    # check if the game is already stored
    game_path = os.path.join(GAME_STORAGE_DIR, f"{game_name}.npz")
    name_file = os.path.join(GAME_STORAGE_DIR, f"{benchmark_name}.names")
    if os.path.exists(game_path) and os.path.exists(name_file):
        game = shapiq.Game(path_to_values=game_path, verbose=verbose, normalize=normalize_loaded)
        player_names = open(name_file).read().splitlines()
        print(f"Loaded game from {game_path}.")
        return game, game_name, player_names

    # set up the game from parameters
    if game_type == "universal":
        game = UniversalHyperparameterImportanceGame(
            benchmark,
            metric=metric,
            n_configs=n_configs,
            random_state=random_state,
            verbose=verbose,
        )
    elif game_type == "global":
        benchmark.set_instance(instance_index)
        game = GlobalHyperparameterImportanceGame(
            benchmark,
            metric=metric,
            n_configs=n_configs,
            random_state=random_state,
            verbose=verbose,
        )
    elif game_type == "local":
        benchmark.set_instance(instance_index)
        optimal_cfg, _ = _find_optimal_configuration(
            benchmark, metric=metric, random_state=random_state, n_configs=n_configs
        )
        game = LocalHyperparameterImportanceGame(benchmark, metric, optimal_cfg, verbose=verbose)
    elif game_type == "universal-local":
        optimal_cfg_list = []
        for instance_index in tqdm(benchmark.instances):
            benchmark.set_instance(instance_index)
            optimal_cfg, _ = _find_optimal_configuration(
                benchmark, metric=metric, random_state=random_state, n_configs=n_configs
            )
            optimal_cfg_list.append(optimal_cfg)
        game = UniversalLocalHyperparameterImportanceGame(
            benchmark, metric, optimal_cfg_list, verbose=verbose
        )
    else:
        raise ValueError(f"Invalid game type: {game_type}")

    # pre-compute and save the game values
    if pre_compute:
        game.precompute()
        game.save_values(game_path)

    # save the player names
    with open(name_file, "w") as f:
        f.write("\n".join(game.tunable_hyperparameter_names))
    player_names = game.tunable_hyperparameter_names

    return game, game_name, player_names
