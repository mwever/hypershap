"""This module contains functions to plot the SI values for different the different games."""

import os

import shapiq
from matplotlib import pyplot as plt
from utils import (
    APPENDIX_PAPER_PLOTS_DIR,
    JAHS_GAME_STORAGE_DIR,
    PARAMETER_NAMES,
    PD1_GAME_STORAGE_DIR,
    YAHPOGYM_GAME_STORAGE_DIR,
    YAHPOGYM_SENS_GAME_STORAGE_DIR,
    abbreviate_player_names,
    get_circular_layout,
    plot_upset,
)

GAME_STORAGES = {
    "ranger": YAHPOGYM_GAME_STORAGE_DIR,
    "pd1": PD1_GAME_STORAGE_DIR,
    "jahs": JAHS_GAME_STORAGE_DIR,
    "lcbench": YAHPOGYM_GAME_STORAGE_DIR,
    "lcbench_sens": YAHPOGYM_SENS_GAME_STORAGE_DIR,
}


def plot_and_save_interactions(file_name: str, benchmark: str, plot_sv: bool = False) -> None:
    """Plots and saves the interactions in a folder."""
    # get parameter names
    parameters = PARAMETER_NAMES[benchmark]
    parameters = abbreviate_player_names(parameters)

    # get the filepaths
    save_name = file_name.replace(".npz", ".pdf")
    save_folder = os.path.join(APPENDIX_PAPER_PLOTS_DIR, "interactions")
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(GAME_STORAGES[benchmark], file_name)

    hpo_game = shapiq.Game(path_to_values=file_path, normalize=True)
    comp = shapiq.ExactComputer(n_players=hpo_game.n_players, game=hpo_game)
    fsii = comp(index="FSII", order=2)  # Note: set to order 1 for SV
    mi = comp(index="Moebius", order=hpo_game.n_players)
    sv = comp(index="SV", order=1)

    # plot upset plot FSII / MI --------------------------------------------------------------------
    upset_folder = os.path.join(save_folder, "upset_plots")
    os.makedirs(upset_folder, exist_ok=True)

    save_path = os.path.join(upset_folder, "FSII_" + save_name)
    plot_upset(fsii, save_path=save_path, show=True, feature_names=parameters)
    print(f"Saved UpSet plot to {save_path}")
    save_path = os.path.join(upset_folder, "MI_" + save_name)
    plot_upset(mi, save_path=save_path, show=True, feature_names=parameters)
    print(f"Saved UpSet plot to {save_path}")
    if plot_sv:
        save_path = os.path.join(upset_folder, "SV_" + save_name)
        plot_upset(sv, save_path=save_path, show=True, feature_names=parameters)
        print(f"Saved UpSet plot to {save_path}")

    # plot SI graph FSII / MI ----------------------------------------------------------------------
    si_folder = os.path.join(save_folder, "si_graphs")
    os.makedirs(si_folder, exist_ok=True)
    pos = get_circular_layout(n_players=hpo_game.n_players)
    plt.rcParams["font.size"] = 18

    save_path = os.path.join(si_folder, "FSII_" + save_name)
    fsii.plot_si_graph(
        show=False,
        size_factor=3.0,
        feature_names=parameters,
        pos=pos,
        n_interactions=1_000,
        compactness=1e50,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved SI graph to {save_path}")
    plt.show()

    save_path = os.path.join(si_folder, "MI_" + save_name)
    mi.plot_si_graph(
        show=False,
        size_factor=3.0,
        feature_names=parameters,
        pos=pos,
        n_interactions=1_000,
        compactness=1e50,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved SI graph to {save_path}")
    plt.show()

    if plot_sv:
        save_path = os.path.join(si_folder, "SV_" + save_name)
        sv.plot_si_graph(
            show=False,
            size_factor=3.0,
            feature_names=parameters,
            pos=pos,
            n_interactions=1_000,
            compactness=1e50,
        )
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved SI graph to {save_path}")
        plt.show()


if __name__ == "__main__":

    # usage: uncomment the game you want to plot and plot this one by one
    # note: do not plot multiple games at once, as the font size is set globally
    pass

    # PD1 plots ------------------------------------------------------------------------------------
    # plot_and_save_interactions("ablation_pd1_imagenet_resnet_512_default_default_n_configs=10000_random_state=42.npz", benchmark="pd1")
    # plot_and_save_interactions("ablation_pd1_lm1b_transformer_2048_default_default_n_configs=10000_random_state=42.npz", benchmark="pd1")
    # plot_and_save_interactions("ablation_pd1_cifar100_wideresnet_2048_default_default_n_configs=10000_random_state=42.npz", benchmark="pd1")
    # plot_and_save_interactions("ablation_pd1_translatewmt_xformer_64_default_default_n_configs=10000_random_state=42.npz", benchmark="pd1")
    # plot_and_save_interactions("data_specific_tunability_pd1_imagenet_resnet_512_default_default_n_configs=10000_random_state=42.npz", benchmark="pd1")
    # plot_and_save_interactions("data_specific_tunability_pd1_lm1b_transformer_2048_default_default_n_configs=10000_random_state=42.npz", benchmark="pd1")
    # plot_and_save_interactions("data_specific_tunability_pd1_cifar100_wideresnet_2048_default_default_n_configs=10000_random_state=42.npz", benchmark="pd1")
    # plot_and_save_interactions("data_specific_tunability_pd1_translatewmt_xformer_64_default_default_n_configs=10000_random_state=42.npz", benchmark="pd1")

    # JAHS plots -----------------------------------------------------------------------------------
    # plot_and_save_interactions("ablation_jahs_jahs_CIFAR10_default_n_configs=10000_random_state=42.npz", benchmark="jahs")
    # plot_and_save_interactions("ablation_jahs_jahs_FashionMNIST_default_n_configs=10000_random_state=42.npz", benchmark="jahs")
    # plot_and_save_interactions("ablation_jahs_jahs_ColorectalHistology_default_n_configs=10000_random_state=42.npz", benchmark="jahs")
    # plot_and_save_interactions("data_specific_tunability_jahs_jahs_CIFAR10_default_n_configs=10000_random_state=42.npz", benchmark="jahs")
    # plot_and_save_interactions("data_specific_tunability_jahs_jahs_FashionMNIST_default_n_configs=10000_random_state=42.npz", benchmark="jahs")
    # plot_and_save_interactions("data_specific_tunability_jahs_jahs_ColorectalHistology_default_n_configs=1000_random_state=42.npz", benchmark="jahs")

    # LCBENCH yapogym plots ------------------------------------------------------------------------
    # plot_and_save_interactions("tunability_yahpogym_lcbench_None_val_accuracy_n_configs=10000_random_state=42.npz", benchmark="lcbench", plot_sv=True)
    # plot_and_save_interactions("tunability_yahpogym_rbv2_ranger_None_acc_n_configs=10000_random_state=42.npz", benchmark="ranger", plot_sv=True)

    # sensitiviy analysis games --------------------------------------------------------------------
    # plot_and_save_interactions("data_specific_tunability_yahpogym-sense_lcbench_0_val_accuracy_n_configs=10000_random_state=42.npz", benchmark="lcbench_sens", plot_sv=True)
    # plot_and_save_interactions("data_specific_tunability_yahpogym-nonsense_lcbench_0_val_accuracy_n_configs=10000_random_state=42.npz", benchmark="lcbench_sens", plot_sv=True)
    # plot_and_save_interactions("data_specific_tunability_yahpogym-sense_lcbench_1_val_accuracy_n_configs=10000_random_state=42.npz", benchmark="lcbench_sens", plot_sv=True)
    # plot_and_save_interactions("data_specific_tunability_yahpogym-nonsense_lcbench_1_val_accuracy_n_configs=10000_random_state=42.npz", benchmark="lcbench_sens", plot_sv=True)
