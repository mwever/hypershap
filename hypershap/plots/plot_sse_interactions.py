"""This script contains functions to plot FSII and MI plots for the experiment with SMAC's surrogate
model. Note that you need to run `hypershap.precompute.smac_analysis_scenario` first."""

import copy
import os

import matplotlib.pyplot as plt
import shapiq
from utils import (
    APPENDIX_PAPER_PLOTS_DIR,
    MAIN_PAPER_PLOTS_DIR,
    abbreviate_player_names,
    get_min_max_of_interactions,
    plot_si_graph,
)

if __name__ == "__main__":
    data_dir = os.path.join("..", "..", "smac_analysis")
    fsii = False  # set to True for FSII plots else MI plots

    APPENDIX_PAPER_PLOTS_DIR = os.path.join(APPENDIX_PAPER_PLOTS_DIR, "smac_analysis")
    MAIN_PAPER_PLOTS_DIR = os.path.join(MAIN_PAPER_PLOTS_DIR, "smac_analysis")
    os.makedirs(APPENDIX_PAPER_PLOTS_DIR, exist_ok=True)
    os.makedirs(MAIN_PAPER_PLOTS_DIR, exist_ok=True)

    parameter_names = [
        "batch_size",
        "learning_rate",
        "max_dropout",
        "max_units",
        "momentum",
        "num_layers",
        "weight_decay",
    ]
    parameter_names = abbreviate_player_names(parameter_names)

    all_interactions, all_games = {}, {}
    for file_name in os.listdir(data_dir):
        if not file_name.endswith(".npz"):
            continue

        file_path = os.path.join(data_dir, file_name)
        budget = int(file_name[:-4].split("_")[-1])

        # get game
        hpo_game = shapiq.Game(path_to_values=file_path, verbose=False, normalize=False)

        # get interaction values
        shap = shapiq.ExactComputer(n_players=hpo_game.n_players, game=hpo_game)
        if fsii:
            interaction = shap(index="FSII", order=2)
        else:
            interaction = shap("Moebius", order=hpo_game.n_players)

        for k, v in interaction.interaction_lookup.items():
            output = ""
            for p in k:
                output += f"{parameter_names[p]} "
            print("|", output, "|", interaction.values[v], "|")

        print("Empty:", hpo_game.exact_values)
        print("Grand:", hpo_game.grand_coalition_value)

        interaction = interaction.get_n_order(hpo_game.n_players, min_order=1)
        all_interactions[budget] = copy.deepcopy(interaction)
        all_games[budget] = copy.deepcopy(hpo_game)

    # get the min and max to scale the plot to the same range
    interactions_list = [int for int in all_interactions.values()]
    min_int, max_int = get_min_max_of_interactions(interactions_list)

    summary = []
    for budget in sorted(all_games.keys()):
        print("Budget", budget)
        interaction = all_interactions[budget]
        game = all_games[budget]
        print(interaction)
        print("Sum", sum(interaction.values))
        print("Empty", game.empty_coalition_value)
        print("Grand", game.grand_coalition_value)

        plt.rcParams.update({"font.size": 18})
        plot = plot_si_graph(
            interaction_values=interaction,
            player_names=parameter_names,
            min_max_interactions=(min_int, max_int),  # scales the plots together
            orders_to_plot=None,  # optionally specify which orders to plot as a list[int]
            show=False,
        )
        fig, ax = plot
        if fsii:
            save_path = os.path.join(APPENDIX_PAPER_PLOTS_DIR, f"fsii_graph_{budget}.pdf")
        else:
            save_path = os.path.join(APPENDIX_PAPER_PLOTS_DIR, f"mi_graph_{budget}.pdf")
        plt.savefig(save_path)
        plt.show()

        summary.append(
            {
                "budget": budget,
                "grand_coalition": game.grand_coalition_value,
                "empty_coalition": game.empty_coalition_value,
                "sum": sum(interaction.values),
            }
        )

    for s in summary:
        print(s)
