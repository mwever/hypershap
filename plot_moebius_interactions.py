import matplotlib
from matplotlib import pyplot as plt
from yahpo_gym import benchmark_set, local_config

from plot_utils import plot_si_graph
from shapiq import ExactComputer
from utils import setup_game


def evaluate_scenario(benchmark, metric, instance_idx=1, hpo_budget=10_000):
    bench = benchmark_set.BenchmarkSet(benchmark)
    instance = bench.instances[instance_idx]
    game, _, param_names = setup_game(
        "global",
        benchmark,
        metric,
        pre_compute=False,
        n_configs=hpo_budget,
        instance_index=instance_idx,
    )

    shap = ExactComputer(n_players=game.n_players, game=game)
    res = shap(index="k-SII", order=7)

    print(instance)
    top_2 = res.get_n_order(order=1).get_top_k(k=2)
    print(top_2.dict_values.keys())
    print(param_names)
    matplotlib.use("TkAgg")
    abbr_param_names = list()
    for p in param_names:
        if "_" in p:
            abbr_param_names.append(("-".join([x[0] for x in p.split("_")])).upper())
        else:
            abbr_param_names.append(p[0].upper())

    plt.rcParams["font.size"] = 18
    plot_si_graph(res, player_names=abbr_param_names)
    plt.savefig(
        "plots/interaction/"
        + (
            "_".join(
                ["SI-Graph", "MI", "global", benchmark, str(instance), str(hpo_budget), metric]
            )
        )
        + ".pdf"
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    local_config.init_config()
    local_config.set_data_path("yahpodata")
    precis_list = [10]
    benchmark = "lcbench"
    metric = "val_accuracy"
    instance_idx = 0

    for instance_idx in range(34):
        evaluate_scenario(benchmark, metric, instance_idx)
