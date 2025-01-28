"""This module contains functions to plot the SI values for different the different games."""

import matplotlib.pyplot as plt
from utils import multiple_si

if __name__ == "__main__":

    # plot params
    plt.rcParams["font.size"] = 18
    SAVE = True
    SHOW = True

    # main -----------------------------------------------------------------------------------------
    multiple_si("global", [0, 1], adjust_min_max=False, show=SHOW, save=SAVE)

    # appendix -------------------------------------------------------------------------------------
    multiple_si("global", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], show=SHOW, save=SAVE)

    # appendix lcbench singles ---------------------------------------------------------------------
    multiple_si("global", [0], index="Moebius", adjust_min_max=False, show=SHOW, save=SAVE)
    multiple_si("global", [0], index="2-FSII", adjust_min_max=False, show=SHOW, save=SAVE)

    multiple_si(
        "local", [1], index="Moebius", adjust_min_max=False, n_configs=100_000, show=SHOW, save=SAVE
    )
    multiple_si(
        "local", [1], index="2-FSII", adjust_min_max=False, n_configs=100_000, show=SHOW, save=SAVE
    )

    multiple_si("universal", [0], index="Moebius", adjust_min_max=False, show=SHOW, save=SAVE)
    multiple_si("universal", [0], index="2-FSII", adjust_min_max=False, show=SHOW, save=SAVE)

    # appendix ranger singles ----------------------------------------------------------------------
    multiple_si(
        "global",
        [0],
        "acc",
        benchmark_name="rbv2_ranger",
        index="Moebius",
        adjust_min_max=False,
        show=SHOW,
        save=SAVE,
    )
    multiple_si(
        "global",
        [0],
        "acc",
        benchmark_name="rbv2_ranger",
        index="2-FSII",
        adjust_min_max=False,
        show=SHOW,
        save=SAVE,
    )

    multiple_si(
        "local",
        [1],
        "acc",
        benchmark_name="rbv2_ranger",
        index="Moebius",
        adjust_min_max=False,
        n_configs=100_000,
        show=SHOW,
        save=SAVE,
    )
    multiple_si(
        "local",
        [1],
        "acc",
        benchmark_name="rbv2_ranger",
        index="2-FSII",
        adjust_min_max=False,
        n_configs=100_00,
        show=SHOW,
        save=SAVE,
    )

    multiple_si(
        "universal",
        [1],
        "acc",
        benchmark_name="rbv2_ranger",
        index="Moebius",
        adjust_min_max=False,
        show=SHOW,
        save=SAVE,
    )
    multiple_si(
        "universal",
        [1],
        "acc",
        benchmark_name="rbv2_ranger",
        index="2-FSII",
        adjust_min_max=False,
        show=SHOW,
        save=SAVE,
    )
