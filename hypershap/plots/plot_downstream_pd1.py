"""Thhis script is for creating the plots of the downstream tasks."""

import os

from utils import plot_downstream

if __name__ == "__main__":

    data_path = "hpo_storage"

    data_id, parameter = "pd1-cifar100_wideresnet_2048", "lr_initial-opt_momentum"
    save_path = "../../downstream_plots"
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"downstream-{data_id}.pdf")
    plot_downstream(data_path, data_id, parameter, save_path)

    data_id, parameter = "pd1-translatewmt_xformer_64", "lr_initial-opt_momentum"
    save_path = "../../downstream_plots"
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"downstream-{data_id}.pdf")
    plot_downstream(data_path, data_id, parameter, save_path)

    data_id, parameter = "pd1-imagenet_resnet_512", "lr_initial-opt_momentum"
    save_path = "../../downstream_plots"
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"downstream-{data_id}.pdf")
    plot_downstream(data_path, data_id, parameter, save_path)

    data_id, parameter = "pd1-lm1b_transformer_2048", "lr_initial-opt_momentum"
    save_path = "../../downstream_plots"
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"downstream-{data_id}.pdf")
    plot_downstream(data_path, data_id, parameter, save_path)
