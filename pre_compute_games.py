"""This example script showcases the different games."""

import time

from yahpo_gym import local_config

import shapiq
from utils import setup_game


def _print_game_info(game: shapiq.Game, setup_time: float) -> None:
    print(
        f"   {game.game_name}",
        f"\n      No. Players: {game.n_players}",
        f"\n      Normalization value: {game.normalization_value}",
        f"\n      Setup time: {setup_time:.6f} seconds",
    )


def inspect_games(
    benchmark_names: list[str],
    game_types: list[str] = None,
    metric: str = "acc",
    pre_compute: bool = False,
    verbose: bool = False,
) -> None:
    """Inspect the different games.

    Args:
        benchmark_names: The benchmark names to inspect.
        game_types: The game types to inspect. Default is `None`.
        metric: The metric to optimize. Default is `"acc"`.
        pre_compute: Whether to pre-compute the game. Default is `False`.
        verbose: Whether to print additional information. Default is `False`.
    """

    if game_types is None:
        game_types = ["universal", "global", "local", "universal-local"]

    for benchmark_name in benchmark_names:
        print(f"\nInspecting benchmark: {benchmark_name}.")

        # Universal game
        if "universal" in game_types:
            setup_time = time.time()
            game_universal, _, _ = setup_game(
                "universal", benchmark_name, metric=metric, pre_compute=pre_compute, verbose=verbose
            )
            setup_time = time.time() - setup_time
            _print_game_info(game_universal, setup_time)

        # Global game
        if "global" in game_types:
            setup_time = time.time()
            game_global, _, _ = setup_game(
                "global", benchmark_name, metric=metric, pre_compute=pre_compute, verbose=verbose
            )
            setup_time = time.time() - setup_time
            _print_game_info(game_global, setup_time)

        # Local game
        if "local" in game_types:
            setup_time = time.time()
            game_local, _, _ = setup_game(
                "local", benchmark_name, metric=metric, pre_compute=pre_compute, verbose=verbose
            )
            setup_time = time.time() - setup_time
            _print_game_info(game_local, setup_time)

        # Universal local game
        if "universal-local" in game_types:
            setup_time = time.time()
            game_universal_local, _, _ = setup_game(
                "universal-local",
                benchmark_name,
                metric=metric,
                pre_compute=pre_compute,
                verbose=verbose,
            )
            setup_time = time.time() - setup_time
            _print_game_info(game_universal_local, setup_time)


if __name__ == "__main__":
    local_config.init_config()
    local_config.set_data_path("yahpodata")

    # "rbv2_super" is excluded due to its long run-time
    benchmark_list = [
        "rbv2_svm",
        "rbv2_rpart",
        "rbv2_aknn",
        "rbv2_glmnet",
        "rbv2_ranger",
        "rbv2_xgboost",
    ]

    hpo_settings = [
        "local",
        "global",
        # "universal",
        "universal-local",
    ]

    inspect_games(
        benchmark_list, game_types=hpo_settings, metric="acc", pre_compute=True, verbose=True
    )
