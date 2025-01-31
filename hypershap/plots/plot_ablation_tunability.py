"""This script is used to generate the paper plot for the comparison of the ablation and tunability
game."""

import os

import matplotlib.pyplot as plt
import shapiq
from utils import (
    APPENDIX_PAPER_PLOTS_DIR,
    MAIN_PAPER_PLOTS_DIR,
    PARAMETER_NAMES,
    abbreviate_player_names,
    get_circular_layout,
    plot_upset,
)

if __name__ == "__main__":

    APPENDIX_PAPER_PLOTS_DIR = os.path.join(APPENDIX_PAPER_PLOTS_DIR, "exp_a")
    os.makedirs(APPENDIX_PAPER_PLOTS_DIR, exist_ok=True)
    MAIN_PAPER_PLOTS_DIR = os.path.join(MAIN_PAPER_PLOTS_DIR, "exp_a")
    os.makedirs(MAIN_PAPER_PLOTS_DIR, exist_ok=True)

    param_names = PARAMETER_NAMES["pd1"]
    param_names = abbreviate_player_names(param_names)

    # get paths
    res_dir = os.path.join("..", "..", "res")
    pd1_dir = os.path.join(res_dir, "games", "pd1")
    ablation_path = os.path.join(
        pd1_dir,
        "ablation_pd1_lm1b_transformer_2048_default_default_n_configs=10000_random_state=42.npz",
    )
    tunability_path = os.path.join(
        pd1_dir,
        "data_specific_tunability_pd1_lm1b_transformer_2048_default_default_n_configs=10000_random_state=42.npz",
    )

    # load games
    game_abl = shapiq.Game(path_to_values=ablation_path, normalize=True)
    game_tun = shapiq.Game(path_to_values=tunability_path, normalize=True)
    n_players = game_abl.n_players
    assert game_abl.n_players == game_tun.n_players == n_players
    pos = get_circular_layout(n_players=n_players)

    # compute values for ablation
    comp_abl = shapiq.ExactComputer(game=game_abl, n_players=n_players)
    mi_abl = comp_abl(index="Moebius", order=n_players)
    fsii_abl = comp_abl(index="FSII", order=2)

    # compute values for tunability
    comp_tun = shapiq.ExactComputer(game=game_tun, n_players=n_players)
    mi_tun = comp_tun(index="Moebius", order=n_players)
    fsii_tun = comp_tun(index="FSII", order=2)

    # Main Paper -----------------------------------------------------------------------------------
    plot_upset(
        mi_abl,
        show=True,
        n_interactions=3,
        feature_names=param_names,
        figsize=(5, 7),
        save_path=os.path.join(MAIN_PAPER_PLOTS_DIR, "ablation_upset.pdf"),
        add_zero_y_lim=True,
    )

    plot_upset(
        mi_tun,
        show=True,
        n_interactions=3,
        feature_names=param_names,
        figsize=(5, 7),
        save_path=os.path.join(MAIN_PAPER_PLOTS_DIR, "tunability_upset.pdf"),
    )

    # Appendix (more information) ------------------------------------------------------------------
    plot_upset(
        mi_abl,
        show=True,
        n_interactions=10,
        feature_names=param_names,
        save_path=os.path.join(APPENDIX_PAPER_PLOTS_DIR, "ablation_upset.pdf"),
        add_zero_y_lim=True,
    )

    plot_upset(
        mi_tun,
        show=True,
        n_interactions=10,
        feature_names=param_names,
        save_path=os.path.join(APPENDIX_PAPER_PLOTS_DIR, "tunability_upset.pdf"),
    )

    # plot si graph plots
    # first increase font size in plt.rcParams
    plt.rcParams["font.size"] = 18
    mi_abl.plot_si_graph(show=False, size_factor=3.0, feature_names=param_names, pos=pos)
    plt.tight_layout()
    plt.savefig(os.path.join(APPENDIX_PAPER_PLOTS_DIR, "ablation_si_graph.pdf"))
    plt.show()
    mi_tun.plot_si_graph(show=False, size_factor=3.0, feature_names=param_names, pos=pos)
    plt.tight_layout()
    plt.savefig(os.path.join(APPENDIX_PAPER_PLOTS_DIR, "tunability_si_graph.pdf"))
    plt.show()
