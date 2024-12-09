"""Thhis script is for creating the plots of the downstream tasks."""

import os

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"hpi_bo": "#7DCE82", "full_bo": "#a8a8a8", "hpi_rs": "#7DCE82", "full_rs": "#a8a8a8"}
MARKERS = {"hpi_bo": "o", "full_bo": "o", "hpi_rs": "s", "full_rs": "s"}
LINESTYLES = {"hpi": "-", "full": "--"}
MARKERS_ON = [0, 1, 2, 3, 4, 9, 19, 49, 99]


def plot_downstream(
    data_id: int, parameter: str, save_path: str, figsize: tuple = (4.5, 4.2)
) -> None:

    fig, axis = plt.subplots(1, 1, figsize=figsize)

    font_size = 14

    # increase the font size
    plt.rcParams.update({"font.size": 12})

    rs_file_name = f"data-{data_id}-False-100-1000-{parameter}.json"
    bo_file_name = f"data-{data_id}-True-100-30-{parameter}.json"

    print(rs_file_name, bo_file_name)

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

    for setting, setting_data in data.items():
        setting_name, setting_type = setting
        y = setting_data["y"]
        yerr = setting_data["yerr"]
        axis.plot(
            x_range,
            y,
            color=COLORS[f"{setting_name}_{setting_type}"],
            linestyle=LINESTYLES[setting_name],
            drawstyle="steps-post",
        )
        axis.fill_between(
            x_range,
            y - yerr,
            y + yerr,
            color=COLORS[f"{setting_name}_{setting_type}"],
            alpha=0.3,
            step="post",
        )

    # plot markers ontop of the lines
    for marker in MARKERS_ON:
        for setting, setting_data in data.items():
            setting_name, setting_type = setting
            y = setting_data["y"]
            axis.plot(
                marker + 1,
                y[marker],
                marker=MARKERS[f"{setting_name}_{setting_type}"],
                color=COLORS[f"{setting_name}_{setting_type}"],
                markeredgecolor="black",
                markeredgewidth=0.75,
            )

    # add legend manually
    # axis.plot([], [], color=COLORS["hpi_bo"], linestyle=LINESTYLES["hpi"], label="HPI")
    # axis.plot([], [], color=COLORS["full_bo"], linestyle=LINESTYLES["full"], label="Naive")
    axis.plot(
        [],
        [],
        marker=MARKERS["hpi_bo"],
        color=COLORS["hpi_bo"],
        linestyle=LINESTYLES["hpi"],
        label="SMAC + HPI",
        markeredgecolor="black",
        markeredgewidth=0.75,
    )
    axis.plot(
        [],
        [],
        marker=MARKERS["hpi_rs"],
        color=COLORS["hpi_rs"],
        linestyle=LINESTYLES["hpi"],
        label="RS + HPI",
        markeredgecolor="black",
        markeredgewidth=0.75,
    )
    axis.plot(
        [],
        [],
        marker=MARKERS["full_bo"],
        color=COLORS["full_bo"],
        linestyle=LINESTYLES["full"],
        label="SMAC",
        markeredgecolor="black",
        markeredgewidth=0.75,
    )
    axis.plot(
        [],
        [],
        marker=MARKERS["full_rs"],
        color=COLORS["full_rs"],
        linestyle=LINESTYLES["full"],
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


if __name__ == "__main__":

    data_path = "hpo_storage"

    data_id, parameter = "pd1-cifar100_wideresnet_2048", "lr_initial-opt_momentum"
    save_path = "downstream_plots"
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"downstream-{data_id}.pdf")
    plot_downstream(data_id, parameter, save_path)

    data_id, parameter = "pd1-translatewmt_xformer_64", "lr_initial-opt_momentum"
    save_path = "downstream_plots"
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"downstream-{data_id}.pdf")
    plot_downstream(data_id, parameter, save_path)

    data_id, parameter = "pd1-imagenet_resnet_512", "lr_initial-opt_momentum"
    save_path = "downstream_plots"
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"downstream-{data_id}.pdf")
    plot_downstream(data_id, parameter, save_path)

    data_id, parameter = "pd1-lm1b_transformer_2048", "lr_initial-opt_momentum"
    save_path = "downstream_plots"
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"downstream-{data_id}.pdf")
    plot_downstream(data_id, parameter, save_path)

