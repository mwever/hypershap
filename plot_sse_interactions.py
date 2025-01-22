"""This module contains functions to plot the SI values for different the different games."""

import copy
import os

import matplotlib.pyplot as plt
import shapiq

from plot_utils import PLOT_DIR, abbreviate_player_names, get_min_max_of_interactions, plot_si_graph

if __name__ == "__main__":
    scenario = "lcbench"
    dataset = 0

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

    # budgets = [25, 50, 100, 200, 400, 800, 1600, 3200]
    budgets = [25, 50, 100, 200]
    interactions_list = []
    for budget in budgets:

        # get game
        path = "_".join(["continuous_smac_analysis_test", scenario, str(dataset), str(budget)])
        path += ".npz"
        hpo_game = shapiq.Game(path_to_values=path, verbose=False, normalize=True)

        # get interaction values
        shap = shapiq.ExactComputer(n_players=hpo_game.n_players, game=hpo_game)
        res = shap(index="k-SII", order=hpo_game.n_players)

        for k, v in res.interaction_lookup.items():
            output = ""
            for p in k:
                output += f"{parameter_names[p]} "
            print("|", output, "|", res.values[v], "|")

        print("Empty:", hpo_game.exact_values)
        print("Grand:", hpo_game.grand_coalition_value)

        interactions_list.append(copy.deepcopy(res))

    # plot the si graphs for the list of interactions

    # get the min and max to scale the plot to the same range
    min_int, max_int = get_min_max_of_interactions(interactions_list)

    save_dir = os.path.join(PLOT_DIR, "si_graphs")
    os.makedirs(save_dir, exist_ok=True)

    for interaction, budget in zip(interactions_list, budgets):
        print(interaction)
        plot = plot_si_graph(
            interaction_values=interaction,
            player_names=parameter_names,
            min_max_interactions=(min_int, max_int),  # scales the plots together
            orders_to_plot=None,  # optionally specify which orders to plot as a list[int]
            show=True,
        )
        if plot is not None:
            fig, ax = plot
            save_path = os.path.join(save_dir, f"si_graph_{scenario}_{dataset}_{budget}.pdf")
            plt.savefig(save_path)
