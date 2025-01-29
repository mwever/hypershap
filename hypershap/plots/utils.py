"""This utils contains functions for plotting"""

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

# define the game storage directories
GAME_STORAGE_DIR = os.path.join("..", "..", "res", "games")
PD1_GAME_STORAGE_DIR = os.path.join(GAME_STORAGE_DIR, "pd1")
if not os.path.exists(PD1_GAME_STORAGE_DIR):
    raise FileNotFoundError(f"PD1 game storage directory not found at {PD1_GAME_STORAGE_DIR}")
YAHPOGYM_GAME_STORAGE_DIR = os.path.join(GAME_STORAGE_DIR, "yahpogym")
if not os.path.exists(YAHPOGYM_GAME_STORAGE_DIR):
    raise FileNotFoundError(
        f"YAHPOGYM game storage directory not found at {YAHPOGYM_GAME_STORAGE_DIR}"
    )
JAHS_GAME_STORAGE_DIR = os.path.join(GAME_STORAGE_DIR, "jahs")
if not os.path.exists(JAHS_GAME_STORAGE_DIR):
    raise FileNotFoundError(f"JAHS game storage directory not found at {JAHS_GAME_STORAGE_DIR}")

PARAMETER_NAMES = {
    "pd1": ["lr_decay_factor", "lr_initial", "lr_power", "opt_momentum"],
    "jahs": [
        "Activation",
        "LearningRate",
        "Op1",
        "Op2",
        "Op3",
        "Op4",
        "Op5",
        "Op6",
        "TrivialAugment",
        "WeightDecay",
    ],
    "ranger": [
        "min.node.size",
        "mtry.power",
        "num.impute.selected.cpo",
        "num.trees",
        "respect.unordered.factors",
        "sample.fraction",
        "splitrule",
        "num.random.splits",
    ],
    "lcbench": [
        "batch_size",
        "learning_rate",
        "max_dropout",
        "max_units",
        "momentum",
        "num_layers",
        "weight_decay",
    ],
}


def plot_upset(
    interactions: shapiq.InteractionValues,
    figsize: tuple[float, float] | tuple[int, int] = None,
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
