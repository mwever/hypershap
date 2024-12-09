"""This module contains functions to plot the SI values for different the different games."""

import copy
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import shapiq
from constants import ABLATION_GAME, DS_TUNABILITY_GAME
from hpo_benchmarks import PD1Benchmark, JAHSBenchmark
from shapiq import ExactComputer
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
        if len(player_name) <= 3:
            abbreviated_player_names.append(player_name)
            continue

        if "_" in player_name:
            abbreviated_player_names.append(
                ("-".join([x[0] for x in player_name.split("_")])).upper()[:3]
            )
        elif "." in player_name:
            abbreviated_player_names.append(
                ("-".join([x[0] for x in player_name.split(".")])).upper()[:3]
            )
        else:
            abbreviated_player_names.append(player_name[0].upper())

    return abbreviated_player_names


def plot_si_graph(
    interaction_values: shapiq.InteractionValues,
    player_names: list[str],
    min_max_interactions: Optional[tuple[float, float]] = None,
    orders_to_plot: Optional[list[int]] = None,
) -> None:
    """Draws a SI-Graph for the given interaction values.

    Args:
        interaction_values: The interaction values to plot.
        player_names: The names of the players.
        min_max_interactions: The minimum and maximum interaction values. Default is `None`. If this
            parameter is not `None`, the plot will be adjusted to these bounds. I.e., all
            interactions will be scaled as if these bounds were the minimum and maximum values. This
            makes it possible to compare different plots more easily.
        orders_to_plot: The orders to plot. Default is `None`. If not `None`, only the specified
            orders will be plotted. Note that the interactions are only removed from the plot, all
            scaling and layouting is still done with the full set of interactions.
    """
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
        node_size_scaling=1.75,  # how big the nodes of the graph are
        compactness=1e50,  # para. for layouting the "explanations" -> higher values centered
        label_mapping=label_mapping,
        circular_layout=True,
        draw_original_edges=False,
        node_area_scaling=False,
        min_max_interactions=min_max_interactions,  # scales the plot to these upper/lower bounds
        orders_to_plot=orders_to_plot,
    )
    plt.tight_layout()


def _convert_si_to_one_dimension(
    interactions: shapiq.InteractionValues, summary_order: Optional[int] = None
) -> tuple[dict[int, np.array], dict[int, np.array]]:
    """Converts the n-Shapley values to one dimension

    Args:
        interactions: The n-Shapley values
        summary_order: The order of the Shapley values. Defaults to the maximum order.

    Returns:
        The positive and negative one-dimensional Shapley values.
    """
    if summary_order is None:
        summary_order = interactions.max_order
    n_players = interactions.n_players

    result_pos = {order: np.zeros(n_players) for order in range(1, summary_order + 1)}
    result_neg = {order: np.zeros(n_players) for order in range(1, summary_order + 1)}

    for S in shapiq.powerset(set(range(n_players)), min_size=1, max_size=summary_order):
        interaction_score = interactions[tuple(S)]
        for player in S:
            if interaction_score > 0:
                result_pos[len(S)][player] += interaction_score / len(S)
            if interaction_score < 0:
                result_neg[len(S)][player] += interaction_score / len(S)
    return result_pos, result_neg


def plot_stacked_bars(
    interactions: shapiq.InteractionValues, feature_names: Optional[list[str]] = None
) -> None:
    """Plots the stacked bar plot for the interactions.

    Args:
        interactions: The interaction values to plot.
        feature_names: The names of the features. Default is `None`.
    """
    pos_values, neg_values = _convert_si_to_one_dimension(interactions)

    title = f"{interactions.index} values up to order {interactions.max_order}"

    _ = shapiq.stacked_bar_plot(
        n_shapley_values_pos=pos_values,
        n_shapley_values_neg=neg_values,
        title=title,
        feature_names=feature_names,
        xlabel="parameters",
        ylabel=f"{interactions.index} values per order",
    )


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
    print(mi)

    # plot the MI as a stacked bar plot
    plot_stacked_bars(mi, feature_names=player_names)
    if save:
        plt.savefig(os.path.join(PLOT_DIR, f"Stacked_MI_{game_name}.pdf"))
    if show:
        plt.show()
    plt.close()

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

    plt.rcParams["font.size"] = 18

    # plot the mi as a si-graph
    plot_si_graph(mi, player_names, min_max_interactions=None)
    if save:
        plt.savefig(os.path.join(PLOT_DIR, f"SI-Graph_MI_{game_name}.pdf"))
    if show:
        plt.show()
    plt.close()

    # plot the two_sii a si-graph
    plot_si_graph(two_sii, player_names)
    if save:
        plt.savefig(os.path.join(PLOT_DIR, f"SI-Graph_2SII_{game_name}.pdf"))
    if show:
        plt.show()
    plt.close()


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
        shap = ExactComputer(n_players=hpo_game.n_players, game_fun=hpo_game)
        res = shap(index="k-SII", order=hpo_game.n_players)

        abbr_player_names = _abbreviate_player_names(parameter_names)
        for k, v in res.interaction_lookup.items():
            output = ""
            for p in k:
                output += f"{abbr_player_names[p]} "
            print("|", output, "|", res.values[v],"|")

        print(hpo_game._lookup_coalitions(np.array([[0,0,0,0]])))
        print(hpo_game._lookup_coalitions(np.array([[1,1,1,1]])))

        # plot_interactions(
        #     game=hpo_game, player_names=parameter_names, game_name=hpo_game_name, save=True, show=True
        # )
