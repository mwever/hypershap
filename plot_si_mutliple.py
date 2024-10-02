"""This module contains functions to plot the SI values for different the different games."""

import os

import matplotlib.pyplot as plt
import numpy as np

import shapiq
from plot_interactions import _abbreviate_player_names, plot_si_graph
from utils import setup_game


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
    game_type,
    instance_indices,
    metric="val_accuracy",
    n_configs=10_000,
    benchmark_name="lcbench",
    index="Moebius",
    adjust_min_max=True,
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
        if index == "Moebius":
            values = computer(index="Moebius", order=hpo_game.n_players)
        elif index == "2-SII":
            values = computer(index="k-SII", order=2)
        elif index == "3-SII":
            values = computer(index="k-SII", order=3)
        elif index == "2-FSII":
            values = computer(index="FSII", order=2)
        elif index == "3-FSII":
            values = computer(index="FSII", order=3)
        else:
            raise ValueError(f"Index {index} not supported.")

        interactions[instance_index] = {
            "interactions": values,
            "game_name": hpo_game_name,
            "player_names": parameter_names,
        }

    scaling = None
    if adjust_min_max:
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
            plt.savefig(os.path.join(PLOT_DIR, f"SI-Graph_{index}_{game_name}.pdf"))
        if SHOW:
            plt.show()
        plt.close()


if __name__ == "__main__":

    PLOT_DIR = "plots/si_multiple"
    os.makedirs(PLOT_DIR, exist_ok=True)

    # plot params
    plt.rcParams["font.size"] = 18
    SAVE = True
    SHOW = True

    # main -----------------------------------------------------------------------------------------
    multiple_si("global", [0, 1], adjust_min_max=False)

    # appendix -------------------------------------------------------------------------------------
    multiple_si("global", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    # appendix lcbench singles ---------------------------------------------------------------------
    multiple_si("global", [0], index="Moebius", adjust_min_max=False)
    multiple_si("global", [0], index="2-FSII", adjust_min_max=False)

    multiple_si("local", [1], index="Moebius", adjust_min_max=False, n_configs=100_000)
    multiple_si("local", [1], index="2-FSII", adjust_min_max=False, n_configs=100_000)

    multiple_si("universal", [0], index="Moebius", adjust_min_max=False)
    multiple_si("universal", [0], index="2-FSII", adjust_min_max=False)

    # appendix ranger singles ----------------------------------------------------------------------
    multiple_si(
        "global", [0], "acc", benchmark_name="rbv2_ranger", index="Moebius", adjust_min_max=False
    )
    multiple_si(
        "global", [0], "acc", benchmark_name="rbv2_ranger", index="2-FSII", adjust_min_max=False
    )

    multiple_si(
        "local",
        [1],
        "acc",
        benchmark_name="rbv2_ranger",
        index="Moebius",
        adjust_min_max=False,
        n_configs=100_000,
    )
    multiple_si(
        "local",
        [1],
        "acc",
        benchmark_name="rbv2_ranger",
        index="2-FSII",
        adjust_min_max=False,
        n_configs=100_00,
    )

    multiple_si(
        "universal", [1], "acc", benchmark_name="rbv2_ranger", index="Moebius", adjust_min_max=False
    )
    multiple_si(
        "universal", [1], "acc", benchmark_name="rbv2_ranger", index="2-FSII", adjust_min_max=False
    )
