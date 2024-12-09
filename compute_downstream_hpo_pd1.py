import json
import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

from downstream_hpo import BOSimulation, RSSimulation
from hpo_benchmarks import PD1Benchmark


def compute_hpo_performance(out_filename, bench, bo, hpo_budget, num_runs, param_sel):
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
            hpoBenchmark=bench,
            hpo_budget=hpo_budget,
            parameter_selection=full_param_set
        )
    else:
        sim_full = RSSimulation(
            hpoBenchmark=bench,
            hpo_budget=hpo_budget,
            parameter_selection=full_param_set
        )

    full_res, full_std = sim_full.simulate_hpo(num_runs=num_runs)
    full_std = full_std / math.sqrt(num_runs)

    print("simulate ", num_runs, " of HPO for the parameter set ", param_sel)
    if bo:
        sim = BOSimulation(
            hpoBenchmark=bench,
            hpo_budget=hpo_budget,
            parameter_selection=param_sel
        )
    else:
        sim = RSSimulation(
            hpoBenchmark=bench,
            hpo_budget=hpo_budget,
            parameter_selection=param_sel
        )

    hpi_res, hpi_std = sim.simulate_hpo(num_runs=num_runs)
    hpi_std = hpi_std / math.sqrt(num_runs)

    data_storage = {
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


def create_step_plot(full_res, full_std, hpi_res, hpi_std, **kwargs):
    # plot the data
    full_res = np.array(full_res)
    full_std = np.array(full_std)
    hpi_res = np.array(hpi_res)
    hpi_std = np.array(hpi_std)
    x = np.arange(1, hpo_budget+1)

    plt.step(x, hpi_res, label="II-HPO")
    plt.fill_between(x, hpi_res - hpi_std, hpi_res + hpi_std, alpha=0.2, step="pre")

    plt.step(x, full_res, label="Plain")
    plt.fill_between(x, full_res - full_std, full_res + full_std, alpha=0.2, step="pre")

    title = "PD1"
    if bo:
        title += " (BO)"
    else:
        title += " (RS)"

    plt.title(title)
    plt.xlabel("Number of Evaluations")
    plt.xscale("log")
    plt.ylabel("Accuracy")

    plt.legend()
    keywords_for_filename = [str("PD1"), "downstream", "hpo", str(num_runs)]
    for k in param_sel:
        keywords_for_filename.append(k)

    if bo:
        keywords_for_filename.append("bo")
    else:
        keywords_for_filename.append("rs")

    plt.savefig("_".join(keywords_for_filename) + ".pdf")
    plt.show()


if __name__ == "__main__":
    for scenario in PD1Benchmark.valid_benchmark_names:
        bench = PD1Benchmark(scenario)
        hpo_budget = 100  # HPO budget
        bo = False
        num_runs = 30 if bo else 1000  # set number of HPO runs for smoother curves

        if os.path.exists("smac3_output/"):
            shutil.rmtree("smac3_output/")

        param_sel = ["lr_initial", "opt_momentum"]

        out_filename = "-".join(
            ["data", "pd1", scenario, str(bo), str(hpo_budget), str(num_runs)] + param_sel
        ) + ".json"
        data = compute_hpo_performance(
            out_filename, bench, bo, hpo_budget, num_runs, param_sel
        )
        create_step_plot(**data)
