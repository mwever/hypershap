from tqdm import tqdm
from yahpo_gym import benchmark_set, local_config
import matplotlib
from shapiq import SHAPIQ, SVARMIQ, KernelSHAPIQ, ExactComputer, network_plot

from hpo_games import (
    UniversalHyperparameterImportanceGame,
    GlobalHyperparameterImportanceGame,
    LocalHyperparameterImportanceGame, UniversalLocalHyperparameterImportanceGame,
)


def evaluate_scenario(benchmark, game_type, metric, approx, precis, instance=None):
    bench = benchmark_set.BenchmarkSet(benchmark)
    if instance is None:
        instance = bench.instances[0]

    print("num datasets", len(bench.instances))

    if game_type == "universal":
        game = UniversalHyperparameterImportanceGame(
            bench, metric, n_configs=1000, random_state=42
        )
    elif game_type == "global":
        bench.set_instance(instance)
        game = GlobalHyperparameterImportanceGame(
            bench, metric, n_configs=1000, random_state=42
        )
    elif game_type == "local":
        bench.set_instance(instance)
        opt_cfg = None
        opt_cfg_value = None
        for cfg in bench.get_opt_space(
            drop_fidelity_params=False, seed=1337
        ).sample_configuration(10000):
            cfg_dict = cfg.get_dictionary()
            cfg_value = bench.objective_function(cfg_dict)[0][metric]
            if opt_cfg_value is None or cfg_value > opt_cfg_value:
                opt_cfg = cfg_dict
                opt_cfg_value = cfg_value
        print("Optimized configuration ", opt_cfg, opt_cfg_value)
        game = LocalHyperparameterImportanceGame(bench, metric, opt_cfg)
    elif game_type == "universallocal":
        opt_cfg_list = list()
        print("compile opt cfg list")
        for instance in tqdm(bench.instances):
            bench.set_instance(instance)
            opt_cfg = None
            opt_cfg_value = None
            for cfg in bench.get_opt_space(
                drop_fidelity_params=False, seed=1337
            ).sample_configuration(10000):
                cfg_dict = cfg.get_dictionary()
                cfg_value = bench.objective_function(cfg_dict)[0][metric]
                if opt_cfg_value is None or cfg_value > opt_cfg_value:
                    opt_cfg = cfg_dict
                    opt_cfg_value = cfg_value
            opt_cfg_list += [opt_cfg]
        game = UniversalLocalHyperparameterImportanceGame(bench, metric, opt_cfg_list)


    approx_cfg = {"n": game.n_players, "random_state": 42}

    if approx == "shapiq":
        shap = SHAPIQ(**approx_cfg)
    elif approx == "svarmiq":
        shap = SVARMIQ(**approx_cfg)
    elif approx == "kerneliq":
        shap = KernelSHAPIQ(**approx_cfg)
    elif approx == "exact":
        shap = ExactComputer(n_players=game.n_players, game_fun=game)
    game.verbose = True
    try:
        res = shap.approximate(budget=precis, game=game)
    except AttributeError:
        res = shap(index="k-SII", order=6)

    value_list = list()
    for k, v in res.interaction_lookup.items():
        param_list = list()
        for i in range(len(k)):
            param_list += [game.tunable_hyperparameter_names[k[i]]]
        value_list += [("+".join(param_list), res.values[v])]
    value_list = sorted(value_list, key=lambda x: x[1])
    for x in value_list:
        print(x)

    matplotlib.use("TkAgg")

    plot = network_plot(
        first_order_values=res.get_n_order_values(1),
        second_order_values=res.get_n_order_values(2),
        feature_names=game.tunable_hyperparameter_names,
    )
    plot[0].savefig(
        "hp_importance_"
        + "_".join([benchmark, game_type, metric, approx, str(precis)])
        + ".png"
    )
    print(
        "stored to file ",
        "hp_importance_"
        + "_".join([benchmark, game_type, metric, approx, str(precis)])
        + ".png",
    )


if __name__ == "__main__":
    local_config.init_config()
    local_config.set_data_path("yahpodata")

    game_types = ["local", "global", "universal", "universallocal"]
    # ["rbv2_svm", "rbv2_rpart", "rbv2_aknn", "rbv2_glmnet", "rbv2_ranger", "rbv2_xgboost", "rbv2_super"]
    game_types = ["universallocal"]
    benchmark_list = ["rbv2_svm"]
    metrics = ["acc"]  # , "bac", "auc", "brier", "f1", "logloss"]
    approx = ["exact"]  # , "svarmiq", "exact"]
    precis_list = [10]

    benchmark_list = ["lcbench"]
    metrics = ["val_accuracy"]
    # game_type = "global"

    for game_type in game_types:
        for metric in metrics:
            for benchmark in benchmark_list:
                for a in approx:
                    for precis in precis_list:
                        print(benchmark, game_type, metric)
                        evaluate_scenario(benchmark, game_type, metric, a, precis)
