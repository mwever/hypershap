import matplotlib
from matplotlib import pyplot as plt
from shapiq import ExactComputer
from utils import get_circular_layout
from yahpo_gym import benchmark_set, local_config

from hypershap.base.util.utils import setup_game


def evaluate_scenario(benchmark, metric, hpo_budget, instance_idx=1):
    bench = benchmark_set.BenchmarkSet(benchmark)
    instance = bench.instances[instance_idx]
    game, _, param_names, _ = setup_game(
        "global",
        benchmark,
        metric,
        pre_compute=True,
        n_configs=hpo_budget,
        instance_index=instance_idx,
    )

    shap = ExactComputer(n_players=game.n_players, game=game)
    res = shap(index="k-SII", order=3)

    matplotlib.use("TkAgg")
    abbr_param_names = list()
    for p in param_names:
        if "_" in p:
            abbr_param_names.append(("-".join([x[0] for x in p.split("_")])).upper())
        else:
            abbr_param_names.append(p[0].upper())

    plt.rcParams["font.size"] = 18
    pos = get_circular_layout(n_players=game.n_players)
    res.plot_si_graph(feature_names=abbr_param_names, pos=pos, size_factor=3.0)
    plt.savefig(
        "plots/hpo_quality/"
        + benchmark
        + "_"
        + str(instance)
        + "_"
        + str(hpo_budget)
        + "_"
        + metric
        + ".png"
    )
    plt.close()


if __name__ == "__main__":
    local_config.init_config()
    local_config.set_data_path("yahpodata")
    precis_list = [10]
    benchmark = "lcbench"
    metric = "val_accuracy"
    instance_idx = 1
    for hpo_budget in [10, 100, 1000, 10000, 100000]:
        print(benchmark, metric, "hpo_budget", hpo_budget)
        evaluate_scenario(benchmark, metric, hpo_budget, instance_idx)
