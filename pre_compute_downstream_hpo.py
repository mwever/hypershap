import json
import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from yahpo_gym import benchmark_set

from downstream_hpo import BOSimulation, RSSimulation


def compute_hpo_performance(out_filename, bench, instance, metric, hpo_budget, num_runs, param_sel):
    path = "hpo_storage/" + out_filename
    if out_filename is not None and os.path.isfile(path):
        return json.load(open(path))

    # determine hyperparameters to be tuned in the full set and the reduced II set.
    full_param_set = list()
    for hp in bench.get_opt_space().get_hyperparameters():
        if hp.name not in ["OpenML_task_id", "epoch"]:
            full_param_set.append(hp.name)

    print("simulate ", num_runs, " of HPO for the full parameter set")
    if bo:
        sim_full = BOSimulation(
            benchmark=bench,
            metric=metric,
            hpo_budget=hpo_budget,
            parameter_selection=full_param_set,
            config_space=bench.get_opt_space(seed=0),
        )
    else:
        sim_full = RSSimulation(
            benchmark=bench,
            metric=metric,
            hpo_budget=hpo_budget,
            parameter_selection=full_param_set,
            config_space=bench.get_opt_space(seed=0),
        )

    full_res, full_std = sim_full.simulate_hpo(num_runs=num_runs)
    full_std = full_std / math.sqrt(num_runs)

    print("simulate ", num_runs, " of HPO for the parameter set ", param_sel)
    if bo:
        sim = BOSimulation(
            benchmark=bench,
            metric=metric,
            hpo_budget=hpo_budget,
            parameter_selection=param_sel,
            config_space=bench.get_opt_space(seed=0),
        )
    else:
        sim = RSSimulation(
            benchmark=bench,
            metric=metric,
            hpo_budget=hpo_budget,
            parameter_selection=param_sel,
            config_space=bench.get_opt_space(seed=0),
        )

    hpi_res, hpi_std = sim.simulate_hpo(num_runs=num_runs)
    hpi_std = hpi_std / math.sqrt(num_runs)

    data_storage = {
        "instance": instance,
        "metric": metric,
        "hpo_budget": hpo_budget,
        "num_runs": num_runs,
        "bo": bo,
        "full_res": full_res.tolist(),
        "full_std": full_std.tolist(),
        "hpi_res": hpi_res.tolist(),
        "hpi_std": hpi_std.tolist(),
    }

    with open(path, "w") as outfile:
        json.dump(data_storage, outfile)

    return data_storage


def create_step_plot(instance, full_res, full_std, hpi_res, hpi_std, **kwargs):
    # plot the data
    full_res = np.array(full_res)
    full_std = np.array(full_std)
    hpi_res = np.array(hpi_res)
    hpi_std = np.array(hpi_std)
    x = np.arange(1, hpo_budget + 1)

    plt.step(x, hpi_res, label="II-HPO")
    plt.fill_between(x, hpi_res - hpi_std, hpi_res + hpi_std, alpha=0.2, step="pre")

    plt.step(x, full_res, label="Plain")
    plt.fill_between(x, full_res - full_std, full_res + full_std, alpha=0.2, step="pre")

    title = str(instance)
    if bo:
        title += " (BO)"
    else:
        title += " (RS)"

    plt.title(title)
    plt.xlabel("Number of Evaluations")
    plt.xscale("log")
    plt.ylabel("Accuracy")

    plt.legend()
    keywords_for_filename = [str(instance), "downstream", "hpo", str(num_runs)]
    for k in param_sel:
        keywords_for_filename.append(k)

    if bo:
        keywords_for_filename.append("bo")
    else:
        keywords_for_filename.append("rs")

    plt.savefig("_".join(keywords_for_filename) + ".pdf")
    plt.show()


if __name__ == "__main__":
    bench = benchmark_set.BenchmarkSet("lcbench")

    metric = "val_accuracy"
    instance_idx = 1
    instance = bench.instances[instance_idx]
    bo = False
    hpo_budget = 100  # HPO budget
    num_runs = 30 if bo else 1000  # set number of HPO runs for smoother curves

    bench.set_instance(instance)

    if os.path.exists("smac3_output/"):
        shutil.rmtree("smac3_output/")

    if instance_idx == 0:
        param_sel = ["batch_size", "num_layers"]
    elif instance_idx == 1:
        param_sel = ["batch_size", "weight_decay"]
    else:
        param_sel = []

    out_filename = "-".join(
        ["data", str(instance), str(bo), str(hpo_budget), str(num_runs)] + param_sel + [".json"]
    )
    data = compute_hpo_performance(
        out_filename, bench, instance, metric, hpo_budget, num_runs, param_sel
    )
    create_step_plot(**data)
