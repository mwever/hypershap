"""Thhis script is for creating the plots of the downstream tasks."""

import os

from utils import plot_downstream

if __name__ == "__main__":

    data_path = "hpo_storage"

    data_id, parameter = 3945, "num_layers"
    save_path = "../../downstream_plots"
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"downstream-{data_id}.pdf")
    plot_downstream(data_path, data_id, parameter, save_path)

    data_id, parameter = 7593, "weight_decay"
    save_path = "../../downstream_plots"
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"downstream-{data_id}.pdf")
    plot_downstream(data_path, data_id, parameter, save_path)
