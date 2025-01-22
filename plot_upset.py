"""This script plots an upset plot for the given interaction values."""

import os

from matplotlib import pyplot as plt

import shapiq
from plot_utils import PLOT_DIR, abbreviate_player_names

if __name__ == "__main__":

    # get the game
    scenario = "lcbench"
    dataset = 0
    budget = 1600
    path = "_".join(["continuous_smac_analysis_test", scenario, str(dataset), str(budget)])
    path += ".npz"
    hpo_game = shapiq.Game(path_to_values=path, verbose=False, normalize=True)

    save_name = "upset_plot_" + scenario + "_" + str(dataset) + "_" + str(budget)
    save_dir = os.path.join(PLOT_DIR, "upset_plots")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    # get parameter names
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

    # get the interaction values
    computer = shapiq.ExactComputer(n_players=hpo_game.n_players, game=hpo_game)
    interactions = computer(index="k-SII", order=hpo_game.n_players)

    # plot the upset plot
    fig = shapiq.upset_plot(
        interaction_values=interactions,
        feature_names=parameter_names,
        n_interactions=15,  # specify how many values to plot
        all_features=True,  # plot all features or only present (some might not be included in n_interactions)
        color_matrix=True,  # set to True for a colored matrix
        show=False,
    )

    # here you can overwrite elements in the axes if you want
    axes = fig.get_axes()
    axis_bar, axis_matrix = axes
    axis_bar.set_ylabel("Interaction Value")

    # finally show or save the plot
    plt.savefig(save_path + ".pdf")
    plt.show()
