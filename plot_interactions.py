"""This module contains functions to plot the SI values for different the different games."""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np

import shapiq
from si_graph import si_graph_plot
from utils import setup_game

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def _abbreviate_player_names(player_names: list[str]) -> list[str]:
    """Abbreviates the player names to be used in the plots.

    Args:
        player_names: The names of the players.

    Returns:
        The abbreviated player names.
    """
    abbreviated_player_names = []
    for player_name in player_names:
        if "." not in player_name:
            abbreviated_player_names.append(player_name[:3] + ".")
        else:
            name_parts = player_name.split(".")
            name = name_parts[0][:1] + "."
            name += name_parts[-1][:3] + "."
            abbreviated_player_names.append(name)

    return abbreviated_player_names


def _plot_si_graph(interaction_values: shapiq.InteractionValues, player_names: list[str]) -> None:
    label_mapping = {i: f"{player_names[i]}" for i in range(interaction_values.n_players)}
    si_graph_nodes = list(
        shapiq.powerset(range(interaction_values.n_players), min_size=2, max_size=2)
    )
    si_graph_interaction = copy.deepcopy(interaction_values)
    try:
        si_graph_interaction.values[si_graph_interaction.interaction_lookup[tuple()]] = 0
    except KeyError:
        pass
    si_graph_plot(
        si_graph_interaction,
        graph=si_graph_nodes,
        size_factor=3,
        node_size_scaling=1.75,
        compactness=1000,
        label_mapping=label_mapping,
        circular_layout=True,
        draw_original_edges=False,
        node_area_scaling=True,
    )
    plt.tight_layout()


def plot_interactions(
    game: shapiq.Game,
    player_names: list[str],
    game_name: str,
    show: bool = True,
    save: bool = False,
) -> None:
    """Plots different visualizations for the SI values.

    Args:
        game: The game to plot the SI values for.
        player_names: The names of the players of the game.
        game_name: The name of the game.
        show: Whether to show the plot. Default is `True`.
        save: Whether to save the plot. Default is `False`.
    """

    player_names = _abbreviate_player_names(player_names)

    # set up the computer and compute the SI values
    computer = shapiq.ExactComputer(n_players=game.n_players, game_fun=game)
    computer.baseline_value = float(game.normalization_value)

    # compute the interactions
    sv: shapiq.InteractionValues = computer(index="SV", order=1)
    two_sii: shapiq.InteractionValues = computer(index="k-SII", order=2)
    mi: shapiq.InteractionValues = computer(index="Moebius", order=game.n_players)

    # plot the SV as a Force Plot
    sv.plot_force(
        feature_names=np.array(player_names),
    )
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, f"Force_SV_{game_name}.pdf"))
    if show:
        plt.show()
    plt.close()

    # plot the two_sii as a Force Plot
    two_sii.plot_force(
        feature_names=np.array(player_names),
    )
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, f"Force_2SII_{game_name}.pdf"))
    if show:
        plt.show()
    plt.close()

    # plot the two_sii as a Network Plot
    two_sii.plot_network(
        feature_names=np.array(player_names),
    )
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, f"Network_2SII_{game_name}.pdf"))
    if show:
        plt.show()
    plt.close()

    # plot the mi as a si-graph
    _plot_si_graph(mi, player_names)
    if save:
        plt.savefig(os.path.join(PLOT_DIR, f"SI-Graph_MI_{game_name}.pdf"))
    if show:
        plt.show()
    plt.close()

    # plot the two_sii a si-graph
    _plot_si_graph(two_sii, player_names)
    if save:
        plt.savefig(os.path.join(PLOT_DIR, f"SI-Graph_2SII_{game_name}.pdf"))
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":

    hpo_game, hpo_game_name, parameter_names = setup_game(
        game_type="universal-local",  # "universal", "global", "local", "universal-local"
        benchmark_name="lcbench",
        metric="val_accuracy",
        pre_compute=False,
        verbose=False,
        n_configs=10_000,
        instance_index=0,
    )
    plot_interactions(
        game=hpo_game, player_names=parameter_names, game_name=hpo_game_name, save=True, show=True
    )
