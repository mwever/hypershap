"""This module contains functions to plot the SI values for different the different games."""

import os

import matplotlib.pyplot as plt
import numpy as np

import shapiq
from plot_interactions import _abbreviate_player_names, plot_si_graph
from utils import setup_game

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def _get_min_max_of_interactions(
    interactions: list[shapiq.InteractionValues],
) -> tuple[float, float]:
    """Computes the minimum and maximum interaction value of a list of InteractionValues.

    Args:
        interactions: The list of InteractionValues to compute the minimum and maximum of.

    Returns:
        The minimum and maximum interaction value.
    """

    min_interactions = [np.min(interaction.values) for interaction in interactions]
    max_interactions = [np.max(interaction.values) for interaction in interactions]

    return float(np.min(min_interactions)), float(np.max(max_interactions))


def multiple_si(
    game_type, instance_indices, metric="val_accuracy", n_configs=10_000, benchmark_name="lcbench"
) -> None:

    interactions = {}
    for instance_index in instance_indices:
        hpo_game, hpo_game_name, parameter_names = setup_game(
            game_type=game_type,
            benchmark_name=benchmark_name,
            metric=metric,
            pre_compute=False,
            verbose=False,
            n_configs=n_configs,
            instance_index=instance_index,
            only_load=True,
        )
        computer = shapiq.ExactComputer(n_players=hpo_game.n_players, game_fun=hpo_game)
        computer.baseline_value = float(hpo_game.normalization_value)
        mi_values = computer(index="Moebius", order=hpo_game.n_players)
        interactions[instance_index] = {
            "interactions": mi_values,
            "game_name": hpo_game_name,
            "player_names": parameter_names,
        }

    scaling = None
    if ADJUST_MIN_MAX:
        interactions_list = [
            interactions[instance_index]["interactions"] for instance_index in instance_indices
        ]
        scaling = _get_min_max_of_interactions(interactions_list)

    for instance_index in instance_indices:
        interaction = interactions[instance_index]["interactions"]
        game_name = interactions[instance_index]["game_name"]
        player_names = _abbreviate_player_names(interactions[instance_index]["player_names"])

        plot_si_graph(interaction, player_names, min_max_interactions=scaling)
        if SAVE:
            plt.savefig(os.path.join(PLOT_DIR, f"SI-Graph_MI_{game_name}.pdf"))
        if SHOW:
            plt.show()
        plt.close()


if __name__ == "__main__":

    # plot params
    plt.rcParams["font.size"] = 18
    SAVE = True
    SHOW = True
    ADJUST_MIN_MAX = True

    # main -----------------------------------------------------------------------------------------
    multiple_si("global", [0, 1])

    # appendix -------------------------------------------------------------------------------------
    multiple_si("global", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
