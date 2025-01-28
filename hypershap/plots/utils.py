"""This utils contains functions for plotting"""

import copy
import os
from typing import Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import shapiq
from matplotlib import pyplot as plt

PAPER_PLOTS_DIR = os.path.join("..", "..", "paper_plots")
MAIN_PAPER_PLOTS_DIR = os.path.join(PAPER_PLOTS_DIR, "main")
APPENDIX_PAPER_PLOTS_DIR = os.path.join(PAPER_PLOTS_DIR, "appendix")
os.makedirs(MAIN_PAPER_PLOTS_DIR, exist_ok=True)
os.makedirs(APPENDIX_PAPER_PLOTS_DIR, exist_ok=True)

PLOT_DIR = os.path.join("..", "..", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


PARAMETER_NAMES = {"pd1": ["L-D", "L-I", "L-P", "O-M"]}


def plot_upset(
    interactions: shapiq.InteractionValues,
    figsize=None,
    save_path: str | None = None,
    add_zero_y_lim: bool = False,
    fontsize_param_names: int | None = 14,
    y_label: str | None = "Hyperparameter Importance",
    **kwargs,
) -> Optional[plt.Figure]:
    """Wrapper function for the upset plot of the interactions."""
    show = True
    if "show" in kwargs:
        show = kwargs["show"]
        kwargs.pop("show")
    fig = interactions.plot_upset(**kwargs, show=False)
    if add_zero_y_lim:  # extract upper axis of the plot and manually set the y-axis to 0 lower lim
        ax = fig.get_axes()[0]
        ax.set_ylim(bottom=-0.001)
    if fontsize_param_names is not None:
        ax = fig.get_axes()[1]
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize_param_names)
        ax.yaxis.label.set_size(fontsize_param_names)
    if y_label is not None:  # add y-label in same fontsize as param names if given
        ax = fig.get_axes()[0]
        if fontsize_param_names is not None:
            ax.yaxis.label.set_size(fontsize_param_names)
        ax.set_ylabel(y_label)
        # also add "parameter" to the y-axis label
        ax = fig.get_axes()[1]
        if fontsize_param_names is not None:
            ax.yaxis.label.set_size(fontsize_param_names)
        ax.set_ylabel("Parameter")
    if figsize is not None:
        fig.set_size_inches(figsize)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if not show:
        return fig
    plt.show()


def get_circular_layout(n_players: int):
    original_graph, graph_nodes = nx.Graph(), []
    for i in range(n_players):
        original_graph.add_node(i, label=i)
        graph_nodes.append(i)
    return nx.circular_layout(original_graph)


def get_min_max_of_interactions(
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


def abbreviate_player_names(player_names: list[str]) -> list[str]:
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
    show=True,
    **kwargs,
) -> Optional[tuple[plt.Figure, plt.Axes]]:
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
        show: Whether to show the plot. Default is `True`.
        **kwargs: Additional keyword arguments to pass to the `si_graph_plot`
    """
    from si_graph import si_graph_plot

    label_mapping = {i: f"{player_names[i]}" for i in range(interaction_values.n_players)}
    si_graph_nodes = list(
        shapiq.powerset(range(interaction_values.n_players), min_size=2, max_size=2)
    )
    si_graph_interaction = copy.deepcopy(interaction_values)
    try:
        si_graph_interaction.values[si_graph_interaction.interaction_lookup[tuple()]] = 0
    except KeyError:
        pass

    if "size_factor" not in kwargs:
        kwargs["size_factor"] = 3
    if "node_size_scaling" not in kwargs:
        kwargs["node_size_scaling"] = 1.75
    if "compactness" not in kwargs:
        kwargs["compactness"] = 1e50
    if "circular_layout" not in kwargs:
        kwargs["circular_layout"] = True
    if "draw_original_edges" not in kwargs:
        kwargs["draw_original_edges"] = False
    if "node_area_scaling" not in kwargs:
        kwargs["node_area_scaling"] = False

    fig, axis = si_graph_plot(
        si_graph_interaction,
        graph=si_graph_nodes,
        min_max_interactions=min_max_interactions,
        orders_to_plot=orders_to_plot,
        label_mapping=label_mapping,
        **kwargs,
    )
    plt.tight_layout()
    if not show:
        return fig, axis
    plt.show()


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
    title = f"{interactions.index} values up to order {interactions.max_order}"

    _ = shapiq.stacked_bar_plot(
        interaction_values=interactions,
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

    player_names = abbreviate_player_names(player_names)

    # set up the computer and compute the SI values
    computer = shapiq.ExactComputer(n_players=game.n_players, game=game)

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


def plot_downstream(
    data_path: str,
    data_id: Union[int, str],
    parameter: str,
    save_path: str,
    figsize: tuple = (4.5, 4.2),
) -> None:

    # settings for the plot
    _colors = {"hpi_bo": "#7DCE82", "full_bo": "#a8a8a8", "hpi_rs": "#7DCE82", "full_rs": "#a8a8a8"}
    _markers = {"hpi_bo": "o", "full_bo": "o", "hpi_rs": "s", "full_rs": "s"}
    _linestyles = {"hpi": "-", "full": "--"}
    _markers_on = [0, 1, 2, 3, 4, 9, 19, 49, 99]
    font_size = 14
    plt.rcParams.update({"font.size": 12})

    # get the data
    rs_file_name = f"data-{data_id}-False-100-1000-batch_size-{parameter}-.json"
    bo_file_name = f"data-{data_id}-True-100-30-batch_size-{parameter}-.json"
    rs_df = pd.read_json(os.path.join(data_path, rs_file_name))
    bo_df = pd.read_json(os.path.join(data_path, bo_file_name))
    data = {
        ("full", "bo"): {
            "y": bo_df["full_res"],
            "yerr": bo_df["full_std"],
        },
        ("full", "rs"): {
            "y": rs_df["full_res"],
            "yerr": rs_df["full_std"],
        },
        ("hpi", "bo"): {
            "y": bo_df["hpi_res"],
            "yerr": bo_df["hpi_std"],
        },
        ("hpi", "rs"): {
            "y": rs_df["hpi_res"],
            "yerr": rs_df["hpi_std"],
        },
    }
    x_range = list(bo_df.index + 1)

    # get the figure
    fig, axis = plt.subplots(1, 1, figsize=figsize)
    for setting, setting_data in data.items():
        setting_name, setting_type = setting
        y = setting_data["y"]
        yerr = setting_data["yerr"]
        axis.plot(
            x_range,
            y,
            color=_colors[f"{setting_name}_{setting_type}"],
            linestyle=_linestyles[setting_name],
            drawstyle="steps-post",
        )
        axis.fill_between(
            x_range,
            y - yerr,
            y + yerr,
            color=_colors[f"{setting_name}_{setting_type}"],
            alpha=0.3,
            step="post",
        )

    # plot markers ontop of the lines
    for marker in _markers_on:
        for setting, setting_data in data.items():
            setting_name, setting_type = setting
            y = setting_data["y"]
            axis.plot(
                marker + 1,
                y[marker],
                marker=_markers[f"{setting_name}_{setting_type}"],
                color=_colors[f"{setting_name}_{setting_type}"],
                markeredgecolor="black",
                markeredgewidth=0.75,
            )

    # add legend manually
    # axis.plot([], [], color=COLORS["hpi_bo"], linestyle=LINESTYLES["hpi"], label="HPI")
    # axis.plot([], [], color=COLORS["full_bo"], linestyle=LINESTYLES["full"], label="Naive")
    axis.plot(
        [],
        [],
        marker=_markers["hpi_bo"],
        color=_colors["hpi_bo"],
        linestyle=_linestyles["hpi"],
        label="SMAC + HPI",
        markeredgecolor="black",
        markeredgewidth=0.75,
    )
    axis.plot(
        [],
        [],
        marker=_markers["hpi_rs"],
        color=_colors["hpi_rs"],
        linestyle=_linestyles["hpi"],
        label="RS + HPI",
        markeredgecolor="black",
        markeredgewidth=0.75,
    )
    axis.plot(
        [],
        [],
        marker=_markers["full_bo"],
        color=_colors["full_bo"],
        linestyle=_linestyles["full"],
        label="SMAC",
        markeredgecolor="black",
        markeredgewidth=0.75,
    )
    axis.plot(
        [],
        [],
        marker=_markers["full_rs"],
        color=_colors["full_rs"],
        linestyle=_linestyles["full"],
        label="RS",
        markeredgecolor="black",
        markeredgewidth=0.75,
    )
    # plot legend with two columns but only little space between the entries
    axis.legend(loc="lower right", ncol=2, columnspacing=0.5, handletextpad=0.5)

    # make x-axis log scale
    axis.set_xscale("log")

    # Set the labels
    axis.set_xlabel("Number of Configurations")
    axis.set_ylabel("Validation Accuracy")
    axis.set_title("HPO Performance")

    axis.set_xlim(0.9, 110)
    if data_id == 3945:
        axis.set_yticks([100, 95, 90, 85, 80])
    if data_id == 7593:
        axis.set_yticks([90, 80, 70, 60, 50])

    # increase font size
    axis.tick_params(axis="both", which="major", labelsize=font_size)
    axis.xaxis.label.set_size(font_size)
    axis.yaxis.label.set_size(font_size)
    axis.title.set_size(font_size)

    # add a grid
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)

    # Show the plot
    plt.show()


def multiple_si(
    game_type,
    instance_indices,
    metric="val_accuracy",
    n_configs=10_000,
    benchmark_name="lcbench",
    index="Moebius",
    adjust_min_max=True,
    save=True,
    show=True,
) -> None:

    from hypershap.base.util.utils import setup_game

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
        computer = shapiq.ExactComputer(n_players=hpo_game.n_players, game=hpo_game)

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
        scaling = get_min_max_of_interactions(interactions_list)

    for instance_index in instance_indices:
        interaction = interactions[instance_index]["interactions"]
        game_name = interactions[instance_index]["game_name"]
        player_names = abbreviate_player_names(interactions[instance_index]["player_names"])

        plot_si_graph(interaction, player_names, min_max_interactions=scaling)
        if save:
            plt.savefig(os.path.join(PLOT_DIR, f"SI-Graph_{index}_{game_name}.pdf"))
        if show:
            plt.show()
        plt.close()
