"""This module contains functions to plot the SI values for different the different games."""

import numpy as np

from constants import DS_TUNABILITY_GAME
from hpo_benchmarks import PD1Benchmark
from plot_utils import abbreviate_player_names, plot_interactions
from shapiq import ExactComputer
from utils import setup_game

if __name__ == "__main__":
    for i in range(4):
        hpo_game, hpo_game_name, parameter_names = setup_game(
            game_type=DS_TUNABILITY_GAME,
            benchmark="pd1",
            scenario=PD1Benchmark.valid_benchmark_names[i],
            metric="default",
            instance_index="default",
            pre_compute=False,
            verbose=False,
            n_configs=10_000,
            only_load=True,
        )
        shap = ExactComputer(n_players=hpo_game.n_players, game=hpo_game)
        res = shap(index="k-SII", order=hpo_game.n_players)

        abbr_player_names = abbreviate_player_names(parameter_names)
        for k, v in res.interaction_lookup.items():
            output = ""
            for p in k:
                output += f"{abbr_player_names[p]} "
            print("|", output, "|", res.values[v], "|")

        print(hpo_game._lookup_coalitions(np.array([[0, 0, 0, 0]])))
        print(hpo_game._lookup_coalitions(np.array([[1, 1, 1, 1]])))

        try:
            plot_interactions(
                game=hpo_game,
                player_names=parameter_names,
                game_name=hpo_game_name,
                save=True,
                show=True,
            )
        except Exception as e:
            print(e)
            pass
