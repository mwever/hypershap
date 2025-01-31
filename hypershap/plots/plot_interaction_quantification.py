"""This module contains function to quantify the level of interactions in the games."""

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import shapiq
from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset
from tqdm import tqdm
from utils import (
    JAHS_GAME_STORAGE_DIR,
    MAIN_PAPER_PLOTS_DIR,
    PD1_GAME_STORAGE_DIR,
    YAHPOGYM_GAME_STORAGE_DIR,
)


def approximated_game(interaction_index):
    n_players = interaction_index.n_players
    grand_coalition_set = set(range(n_players))
    approximation_lookup = {}
    approximation_values = np.zeros(2**n_players)
    for coalition_pos, coalition in enumerate(powerset(grand_coalition_set)):
        approximation_lookup[coalition] = coalition_pos
        for interaction in powerset(coalition):
            approximation_values[coalition_pos] += interaction_index[interaction]

    baseline_value = approximation_values[approximation_lookup[tuple()]]
    if baseline_value != 0:  # we only want to include normalized values
        raise ValueError("Baseline value is not zero.")
    approximation = InteractionValues(
        index=interaction_index.index,
        max_order=n_players,
        n_players=n_players,
        min_order=0,
        baseline_value=float(baseline_value),
        interaction_lookup=approximation_lookup,
        values=approximation_values,
    )
    return approximation


def _convert_game_to_interaction(exact_computer: shapiq.ExactComputer) -> shapiq.InteractionValues:
    """Converts the exact game values to interaction values.

    Args:
        exact_computer: The exact computer to convert the game values from.

    Returns:
        The interaction values.
    """
    game_values = shapiq.InteractionValues(
        values=exact_computer.game_values,
        index="Moebius",
        interaction_lookup=exact_computer.coalition_lookup,
        n_players=exact_computer.n,
        min_order=0,
        max_order=exact_computer.n,
        baseline_value=exact_computer.baseline_value,
    )
    return game_values


def _get_weight(n: int, coalition_size: int, uniform_weights: bool = False) -> float:
    """Computes the weight for the given coalition size.

    Args:
        n: The number of players.
        coalition_size: The size of the coalition.
        uniform_weights: Whether to use uniform weights or a Shapley kernel. Default is `False`
            (Shapley kernel).

    Returns:
        The weight for the given coalition size.
    """
    if uniform_weights:
        return (1 / 2) ** n
    if coalition_size == n or coalition_size == 0:
        sv_weight = 0
    else:
        sv_weight = scipy.special.binom(n, coalition_size) * coalition_size * (n - coalition_size)
        sv_weight = (n - 1) / sv_weight
    return sv_weight


def _get_approximation_weights(n: int, uniform_weights: bool = False) -> np.ndarray:
    """Computes the weights for the approximations.

    Args:
        n: The number of players.
        uniform_weights: Whether to use uniform weights or a Shapley kernel. Default is `False`
            (Shapley kernel).

    Returns:
        The weights for the approximations as a numpy array.
    """
    approximation_weights = np.zeros(2**n, dtype=float)
    grand_coalition_set = set(range(n))
    for coalition_pos, coalition in enumerate(shapiq.powerset(grand_coalition_set)):
        approximation_weights[coalition_pos] = _get_weight(n, len(coalition), uniform_weights)
    return approximation_weights


def get_approximation_error(
    approximations: dict[str, dict[int, shapiq.InteractionValues]],
    game_values: shapiq.InteractionValues,
    uniform_weights: bool = False,
) -> dict[str, dict[int, float]]:
    """Computes the approximation error for the approximated interaction values.

    Args:
        approximations: The approximated interaction values.
        game_values: The exact game values.
        uniform_weights: Whether to use uniform weights or a Shapley kernel. Default is `False`
            (Shapley kernel).

    Returns:
        A dictionary containing the approximation errors for the approximated interaction values.
    """
    weights = _get_approximation_weights(game_values.n_players, uniform_weights=uniform_weights)
    game_values_copy = copy.deepcopy(game_values)
    approximation_errors = {}
    for index, approximations_order in approximations.items():
        approximation_errors[index] = {}
        for order, approximation in approximations_order.items():
            # get the approximation values
            approx = copy.deepcopy(approximations[index][order])
            approx_arr = np.zeros(len(game_values.values))
            approx_arr[: len(approx.values)] = approx.values  # padded with zeros

            # get the difference between the game and si values
            game_values_copy.index = approx.index  # workaround to compare
            difference = approx - game_values_copy  # take the difference between game and si

            # compute the weighted r2 score
            error = np.sum(weights * difference.values**2)
            weighted_average = np.sum(weights * approx_arr) / np.sum(weights)
            total_sum_of_squares = np.sum(weights * (approx_arr - weighted_average) ** 2)
            r2_score = 1 - error / total_sum_of_squares

            approximation_errors[index][order] = r2_score
    return approximation_errors


def evaluate_game(
    game: shapiq.Game, index: str = "FSII", verbose: bool = False
) -> tuple[dict, dict]:
    """Evaluates the game.

    Args:
        game: The game to evaluate.
        index: The indices to approximate the game with. Default is `FSII`.
        verbose: Whether to print verbose output. Default is `False`.

    Returns:
        A tuple containing the approximation errors and the Möbius transform.
    """
    computer = shapiq.ExactComputer(n_players=game.n_players, game=game)
    computer._baseline_value = float(game.normalization_value)
    game_values = _convert_game_to_interaction(computer)

    moebius, approximations = {}, {}
    approximations[index] = {}
    for order in range(1, game.n_players + 1):
        interactions = computer.shapley_interaction(index=index, order=order)
        approximations[index][order] = approximated_game(interactions)
        if order == game.n_players:  # For highest order Shapley interactions are Möbius transform
            moebius[index] = interactions
    approximation_errors = get_approximation_error(approximations, game_values)
    if verbose:
        print("R2 Scores per Order:", approximation_errors)
    return approximation_errors, moebius


def _aggregate_r2(
    r2_scores: list[dict[str, dict[int, float]]], index="FSII"
) -> dict[str, dict[int, float]]:
    import pandas as pd

    scores = []
    for score_game in r2_scores:
        score_orders = score_game[index]
        for order, r2_score in score_orders.items():
            scores.append({"order": order, "r2_score": r2_score})
    r2_scores_df = pd.DataFrame(scores)

    scores_mean, scores_std, scores_sem = {}, {}, {}
    for order in range(1, r2_scores_df["order"].max() + 1):
        scores = r2_scores_df[r2_scores_df["order"] == order]["r2_score"]
        scores_mean[order] = float(scores.mean())
        scores_std[order] = float(scores.std())
        scores_sem[order] = float(scores.sem())
    return {"mean": scores_mean, "std": scores_std, "sem": scores_sem}


COLORS = {
    "universal": "#00b4d8",
    "global": "#ef27a6",
    "local": "#ff6f00",
}

COLORS_LIGHTER = {
    "universal": "#d8f9ff",
    "global": "#fddef1",
    "local": "#ffe9d8",
}

MARKERS = {
    "universal": "D",
    "global": "s",
    "local": "o",
}

LABELS = {
    "universal": "Tunability",
    "global": "Data-Specific Tunability",
    "local": "Ablation",
}


def plot_r2_agg(
    scores: dict[str, dict[str, dict[int, float]]], figsize: tuple = (5, 4)
) -> tuple[plt.Figure, plt.Axes]:
    fig, axis = plt.subplots(1, 1, figsize=figsize)

    for game_type in ["local", "global", "universal"]:
        scores_agg = scores[game_type]
        color = COLORS[game_type]
        label_name = LABELS[game_type]
        marker = MARKERS[game_type]

        orders = list(sorted(scores_agg["mean"].keys()))
        scores_mean = np.array([scores_agg["mean"][order] for order in orders])
        axis.plot(
            orders,
            scores_mean,
            label=label_name,
            color=color,
            marker=marker,
        )
        if "sem" in scores_agg:
            scores_std = np.array([scores_agg["sem"][order] for order in orders])
            axis.fill_between(
                orders,
                scores_mean - scores_std,
                scores_mean + scores_std,
                color=color,
                alpha=0.15,
            )

    axis.set_xticks(range(1, len(orders) + 1))
    axis.set_xlabel("Explanation Order")
    axis.set_ylabel("Shapley-weighted R2")
    axis.set_ylim(-0.025, 1.05)
    axis.legend(loc="lower right")
    axis.set_title("Faithfulness of Explanations")
    axis.grid(True, which="both", linestyle="--", linewidth=0.5)
    axis.tick_params(axis="both", which="both", length=0)
    plt.tight_layout()
    return fig, axis


def plot_violine_agg(
    moebius: dict[str, dict[str, shapiq.InteractionValues]],
    figsize: tuple = (5, 4),
    violin: bool = True,
    colorful: bool = True,
    ylim: tuple = None,
    y_ticks: list = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, axis = plt.subplots(1, 1, figsize=figsize)

    width = 0.25
    x_offsets = [-0.3, 0, 0.3]

    for len_id, game_type in enumerate(["local", "global", "universal"]):

        x_offset = x_offsets[len_id]
        color = COLORS[game_type]

        if not colorful:
            edge_color = "#000000"
            face_color = "#eeeeee"
        else:
            edge_color = color
            face_color = color + "10"

        moebius_vals = list(moebius[game_type].values())
        x_vals, y_vals = [], []
        for moebius_val in moebius_vals:
            for key, val in moebius_val.dict_values.items():
                x_vals.append(len(key)), y_vals.append(val)
        df = pd.DataFrame({"size": x_vals, "interaction": y_vals})
        df = df[df["size"] > 0]
        df["interaction_abs"] = np.abs(df["interaction"])
        unique_x_vals = df["size"].unique()
        positions = sorted(df["size"].unique() + x_offset)
        grouped_data = [df[df["size"] == cat]["interaction_abs"].values for cat in unique_x_vals]
        if violin:
            violin_parts = axis.violinplot(
                grouped_data, positions=positions, showmeans=True, showextrema=False, widths=width
            )
            for i, pc in enumerate(violin_parts["bodies"]):
                pc.set_facecolor(COLORS_LIGHTER[game_type])
                pc.set_edgecolor(edge_color)
                pc.set_linewidth(1)
                pc.set_alpha(1)
            vp = violin_parts["cmeans"]
            vp.set_edgecolor(edge_color)
            vp.set_linewidth(1)
            axis.plot([], [], color=color, label=LABELS[game_type])
        else:  # boxplot
            axis.boxplot(
                grouped_data,
                positions=positions,
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor=face_color, color=edge_color),
                whiskerprops=dict(color=edge_color),
                capprops=dict(color=edge_color),
                medianprops=dict(color=color),
                showfliers=False,
                showmeans=True,
                meanprops=dict(
                    marker=MARKERS[game_type], markerfacecolor=color, markeredgecolor=color
                ),
            )
            axis.plot([], [], color=color, label=LABELS[game_type], marker=MARKERS[game_type])

    axis.set_xticks(range(1, len(unique_x_vals) + 1))
    axis.set_xticklabels(unique_x_vals)

    # add a grey rectangles for each interaction order
    for i in range(1, len(unique_x_vals) + 1):
        if i % 2 == 0:
            continue
        axis.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, 100, color="#eeeeee", alpha=0.5, zorder=0))

    if ylim is not None:
        axis.set_ylim(ylim)
    axis.set_xlim(0.5, len(unique_x_vals) + 0.5)
    if y_ticks is not None:
        axis.set_yticks(y_ticks)
    axis.legend(loc="upper right")
    axis.set_xlabel("Interaction Order")
    axis.set_ylabel("Absolute Interaction Effect")
    axis.set_title("Magnitude of Interactions")
    # add only horizontal grid lines
    axis.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    axis.tick_params(axis="both", which="both", length=0)
    plt.tight_layout()

    return fig, axis


def load_games(
    benchmark_name: str, game_type: str, n_instances: int | None = None
) -> list[shapiq.Game]:
    """Helper function to load a pre-computed game from disc for evaluation."""
    available_game_types = ["ablation", "tunability", "optbias", "data_specific_tunability"]
    assert game_type in available_game_types
    if benchmark_name == "lcbench":
        game_storage_dir = YAHPOGYM_GAME_STORAGE_DIR
    elif benchmark_name.startswith("rbv2"):
        game_storage_dir = YAHPOGYM_GAME_STORAGE_DIR
    elif benchmark_name == "pd1":
        game_storage_dir = PD1_GAME_STORAGE_DIR
    elif benchmark_name == "jahs":
        game_storage_dir = JAHS_GAME_STORAGE_DIR
    else:
        raise ValueError(f"Invalid benchmark name: {benchmark_name}")

    all_files = os.listdir(game_storage_dir)
    all_files = [file for file in all_files if file.endswith(".npz")]
    all_files = [file for file in all_files if benchmark_name in file]

    # select the correct game type
    all_files = [file for file in all_files if game_type + "_" in file]
    if game_type == "tunability":  # remove all data-specific tunability games
        all_files = [file for file in all_files if "data_specific_tunability" not in file]
    all_files = sorted(set(all_files))
    if n_instances is not None:
        all_files = all_files[:n_instances]
    print(f"Found {len(all_files)} games for {game_type} in {benchmark_name}. Using {n_instances}.")

    games = []
    for file in all_files:
        game = shapiq.Game(path_to_values=os.path.join(game_storage_dir, file), normalize=True)
        games.append(game)

    return games


def create_plot_data(
    games_to_plot: list[tuple[str, str, int | None]], index: str = "FSII", only_load: bool = False
) -> pd.DataFrame:
    save_name = "plot_data.csv"

    if only_load:
        return pd.read_csv(save_name)

    # load the games -------------------------------------------------------------------------------
    all_games, n_games = {}, 0
    for benchmark_name, game_type, n_instances in games_to_plot:
        games = load_games(benchmark_name, game_type, n_instances)
        all_games[(benchmark_name, game_type)] = games
        n_games += len(games)
    print(f"Loaded {n_games} games to plot")

    # evaluate the games ---------------------------------------------------------------------------
    all_results = {}
    n_players_max = 0
    pbar = tqdm(total=n_games, desc="Evaluating games")
    for (benchmark_name, game_type), games in all_games.items():
        results = {"errors": [], "moebius": [], "n_games": len(games)}
        for game in games:
            errors, mi = evaluate_game(game, index=index)
            results["errors"].append(errors)
            results["moebius"].append(mi)
            pbar.update(1)
            n_players_max = max(n_players_max, game.n_players)
        all_results[(benchmark_name, game_type)] = results
    pbar.close()
    print(f"Evaluated {n_games} games. Maximum number of players: {n_players_max}")

    # aggregate the results ------------------------------------------------------------------------
    plot_data = []
    for (benchmark_name, game_type), results in all_results.items():
        scores = _aggregate_r2(results["errors"])
        n_games = results["n_games"]
        mean, std, sem = scores["mean"], scores["std"], scores["sem"]
        for order in mean.keys():
            plot_data.append(
                {
                    "benchmark": benchmark_name,
                    "game_type": game_type,
                    "order": order,
                    "mean": mean[order],
                    "std": std[order],
                    "sem": sem[order],
                    "n_games": n_games,
                }
            )

    # save the results to disc
    plot_data = pd.DataFrame(plot_data)
    plot_data.to_csv(save_name, index=False)
    print(f"Saved plot data to {save_name}")

    return plot_data


LEGEND_NAME_MAPPING = {
    "jahs": "JAHS-Bench-201",
    "rbv2_ranger": "rbv2_ranger",
    "lcbench": "lcbench",
    "pd1": "PD1",
    # game types
    "tunability": "Multi-Data Tunability",
    "data_specific_tunability": "Tunability",
    "ablation": "Ablation",
}

MARKER_MAPPING = {
    "jahs": "o",
    "rbv2_ranger": "s",
    "lcbench": "D",
    "pd1": "X",
}

COLOR_MAPPING = {
    "tunability": "#00b4d8",
    "data_specific_tunability": "#ff6f00",
    "ablation": "#ef27a6",
}


if __name__ == "__main__":

    save_folder = os.path.join(MAIN_PAPER_PLOTS_DIR, "interaction_quantification")
    os.makedirs(save_folder, exist_ok=True)

    index: str = "FSII"
    max_order_to_plot: int | None = 5
    only_load: bool = True

    # configure the games to plot
    games_to_plot: list[tuple[str, str, int | None]] = [
        ("jahs", "data_specific_tunability", None),  # order 10
        ("pd1", "data_specific_tunability", None),  # order 4
        ("rbv2_ranger", "data_specific_tunability", 20),  # order 8
        ("lcbench", "data_specific_tunability", 20),  # order 7
        ("rbv2_ranger", "tunability", None),  # order 8
        ("lcbench", "tunability", None),  # order 7
    ]

    legend_order = [
        ("jahs", "data_specific_tunability"),
        ("pd1", "data_specific_tunability"),
        ("rbv2_ranger", "data_specific_tunability"),
        ("lcbench", "data_specific_tunability"),
        ("rbv2_ranger", "tunability"),
        ("lcbench", "tunability"),
    ]

    # load the data --------------------------------------------------------------------------------
    data = create_plot_data(games_to_plot, index=index, only_load=only_load)

    # plot the results -----------------------------------------------------------------------------
    styling = {"mew": 1, "mec": "white", "markersize": 7, "linestyle": "-", "linewidth": 2}

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3))
    plotted_games = set()
    for benchmark_name, game_type, _ in games_to_plot:
        data_subset = data[(data["benchmark"] == benchmark_name) & (data["game_type"] == game_type)]
        if len(data_subset) == 0:
            print(f"Skipping {benchmark_name} {game_type} as no data is available.")
            continue
        order = max(data_subset["order"].values)
        print(f"Adding {benchmark_name} {game_type} with {order} orders.")
        plotted_games.add((benchmark_name, game_type))

        name = LEGEND_NAME_MAPPING[game_type]
        name += f" ({LEGEND_NAME_MAPPING[benchmark_name]})"
        marker = MARKER_MAPPING[benchmark_name]
        color = COLOR_MAPPING[game_type]
        ax.plot(data_subset["order"], data_subset["mean"], marker=marker, color=color, **styling)
        # use sem for error bars
        ax.fill_between(
            data_subset["order"],
            data_subset["mean"] - data_subset["sem"],
            data_subset["mean"] + data_subset["sem"],
            color=color,
            alpha=0.25,
        )

    for benchmark_name, game_type in legend_order:
        if (benchmark_name, game_type) not in plotted_games:
            continue

        name = LEGEND_NAME_MAPPING[game_type]
        name += f" ({LEGEND_NAME_MAPPING[benchmark_name]})"
        marker = MARKER_MAPPING[benchmark_name]
        color = COLOR_MAPPING[game_type]
        ax.plot([], [], label=name, marker=marker, color=color, **styling)

    # beautify the plot
    if max_order_to_plot is not None:
        ax.set_xlim(0.5, max_order_to_plot + 0.5)
    ax.set_ylim(0.45, 1.05)  # ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")

    # increase fontsizes of ticks and axis labels
    ax.tick_params(axis="both", which="both", labelsize=12)
    ax.set_xlabel("Explanation Order", fontsize=12)
    # write R^2 in math mode
    ax.set_ylabel(r"Shapley-weighted $R^2$", fontsize=12)
    # ax.set_title("Faithfulness", fontsize=18)

    # add vertical grid lines
    for i in range(1, max_order_to_plot + 1):
        if i % 2 == 0:
            continue
        ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, 100, color="#eeeeee", alpha=0.5, zorder=0))

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    save_path = os.path.join(save_folder, "faithfulness.pdf")
    print(f"\nSaving plot to {save_path}")
    plt.savefig(save_path)
    plt.show()
