import json
import os

import matplotlib.pyplot as plt
import numpy as np
from utils import APPENDIX_PAPER_PLOTS_DIR, MAIN_PAPER_PLOTS_DIR


def create_step_plot(
    instance: int,
    data: dict,
    save_dir: str,
    show: bool = False,
    add_title: bool = True,
    log_scale: bool = True,
    fig_size=(4, 4),
) -> None | tuple[plt.Figure, plt.Axes]:

    # stylings
    _colors = {"hypershap": "#7DCE82", "fanova": "black", "opt": "#a8a8a8"}
    _labels = {"hypershap": "HyperSHAP (top-2)", "fanova": "fANOVA (top-2)", "opt": "Optimum"}
    _linestyles = {"hypershap": "-", "fanova": "-", "opt": "--"}
    _markers_on = [0, 1, 2, 3, 4, 9, 19, 49, 99]

    styling = {
        "mew": 0.5,
        "mec": "black",
        "markersize": 5,
        "linewidth": 2.5,
    }

    # get data
    full_res = data["fn_res"]
    full_std = data["fn_std"]
    hpi_res = data["hs_res"]
    hpi_std = data["hs_std"]
    opt = data["full_opt"]
    bo = data["bo"]

    # plot the data
    full_res = np.array(full_res)
    full_std = np.array(full_std)
    hpi_res = np.array(hpi_res)
    hpi_std = np.array(hpi_std)
    opt = np.array([float(opt)] * 200)
    opt_value = max(opt)
    x = np.arange(1, 200 + 1)

    font_size = 12
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    # add optimal value
    ax.axhline(
        opt_value,
        color=_colors["opt"],
        linestyle=_linestyles["opt"],
        label=_labels["opt"],
        linewidth=1,
    )

    # add fanova
    ax.step(
        x,
        full_res,
        label=_labels["fanova"],
        color=_colors["fanova"],
        linestyle=_linestyles["fanova"],
        **styling,
    )
    ax.fill_between(
        x, full_res - full_std, full_res + full_std, alpha=0.25, step="pre", color=_colors["fanova"]
    )

    # add hypershap
    ax.step(
        x,
        hpi_res,
        label=_labels["hypershap"],
        color=_colors["hypershap"],
        linestyle=_linestyles["hypershap"],
        **styling,
    )
    ax.fill_between(
        x, hpi_res - hpi_std, hpi_res + hpi_std, alpha=0.25, step="pre", color=_colors["hypershap"]
    )

    title = f"Data ID: {instance}"
    if add_title:
        ax.set_title(title, fontsize=font_size)

    ax.set_xlabel("Number of Evaluations", fontsize=font_size)
    ax.set_ylabel("Accuracy", fontsize=font_size)
    ax.legend(ncols=1)

    if log_scale:
        ax.set_xscale("log")
        ax.set_xlim(0.8, 250)
        x_zero = 0.8
    else:
        ax.set_xlim(-8, 208)
        x_zero = -8

    # draw a special indicator at x_zero for the optimal value ontop of the y-axis
    ax.plot(
        x_zero,
        opt_value,
        color="white",
        marker="D",
        markersize=6,
        linestyle="None",
        zorder=20,
        mec="black",
        mew=1,
    )
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    # only show interger on y-axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # add grid on 1, 10, 100, 1000
    for i in range(4):
        ax.axvline(10**i, color=_colors["opt"], linestyle="dotted", linewidth=0.5)

    plt.tight_layout()

    # save the plot
    keywords_for_filename = [str(instance), "downstream", "hpo", str(1000)]
    if fig_size != (4, 4):
        keywords_for_filename.append("size")
    if bo:
        keywords_for_filename.append("bo")
    else:
        keywords_for_filename.append("rs")
    save_path = os.path.join(save_dir, "_".join(keywords_for_filename) + ".pdf")
    plt.savefig(save_path)

    if not show:
        return fig, ax
    plt.show()


if __name__ == "__main__":

    data_folder = os.path.join("..", "..", "res", "hpo_runs", "fanova")

    # main paper plot ------------------------------------------------------------------------------
    save_dir = os.path.join(MAIN_PAPER_PLOTS_DIR, "comparison_fanova")
    os.makedirs(save_dir, exist_ok=True)

    data_id = 126029
    data_path = os.path.join(data_folder, f"wref_fanova_vs_hypershap_downstream_{data_id}.json")
    create_step_plot(
        data_id,
        data=json.load(open(data_path)),
        save_dir=save_dir,
        show=True,
        fig_size=(6.1, 3.2),
        add_title=False,
    )

    # appendix plots for the paper -----------------------------------------------------------------
    save_dir = os.path.join(APPENDIX_PAPER_PLOTS_DIR, "comparison_fanova")
    os.makedirs(save_dir, exist_ok=True)

    data_id = 126025
    data_path = os.path.join(data_folder, f"wref_fanova_vs_hypershap_downstream_{data_id}.json")
    create_step_plot(data_id, data=json.load(open(data_path)), save_dir=save_dir, show=True)

    data_id = 126026
    data_path = os.path.join(data_folder, f"wref_fanova_vs_hypershap_downstream_{data_id}.json")
    create_step_plot(data_id, data=json.load(open(data_path)), save_dir=save_dir, show=True)

    data_id = 126029
    data_path = os.path.join(data_folder, f"wref_fanova_vs_hypershap_downstream_{data_id}.json")
    create_step_plot(data_id, data=json.load(open(data_path)), save_dir=save_dir, show=True)

    data_id = 167181
    data_path = os.path.join(data_folder, f"wref_fanova_vs_hypershap_downstream_{data_id}.json")
    create_step_plot(data_id, data=json.load(open(data_path)), save_dir=save_dir, show=True)

    data_id = 167190
    data_path = os.path.join(data_folder, f"wref_fanova_vs_hypershap_downstream_{data_id}.json")
    create_step_plot(data_id, data=json.load(open(data_path)), save_dir=save_dir, show=True)

    data_id = 168910
    data_path = os.path.join(data_folder, f"wref_fanova_vs_hypershap_downstream_{data_id}.json")
    create_step_plot(data_id, data=json.load(open(data_path)), save_dir=save_dir, show=True)

    data_id = 189865
    data_path = os.path.join(data_folder, f"wref_fanova_vs_hypershap_downstream_{data_id}.json")
    create_step_plot(data_id, data=json.load(open(data_path)), save_dir=save_dir, show=True)

    data_id = 189866
    data_path = os.path.join(data_folder, f"wref_fanova_vs_hypershap_downstream_{data_id}.json")
    create_step_plot(data_id, data=json.load(open(data_path)), save_dir=save_dir, show=True)
