import matplotlib
from tqdm import tqdm
from constants import ABLATION_GAME, SET_ABLATION_GAME, TUNABILITY_GAME, DS_TUNABILITY_GAME, OPTIMIZER_BIAS_GAME, DS_OPTIMIZER_BIAS_GAME
from hpo_benchmarks import YahpoGymBenchmark

from hpo_games import (
    AblationHPIGame,
    AblationSetHPIGame,
    DataSpecificTunabilityHPIGame,
    TunabilityHPIGame,
    DataSpecificOptimizerBiasGame,
    OptimizerBiasGame,
)
from hpo_optimizers import SubspaceRandomOptimizer, LocalOptimizer
from shapiq import SHAPIQ, SVARMIQ, ExactComputer, KernelSHAPIQ, network_plot


def get_opt_cfg(hpoBenchmark, instance, n_configs=10_000, random_state=None):
    opt_cfg, opt_cfg_value = None, None
    for cfg in hpoBenchmark.sample_configurations(n=n_configs, random_state=random_state):
        cfg_value = hpoBenchmark.evaluate(cfg, instance)
        if opt_cfg_value is None or cfg_value > opt_cfg_value:
            opt_cfg = cfg
            opt_cfg_value = cfg_value
    print("Optimized configuration ", opt_cfg, opt_cfg_value)
    return opt_cfg, opt_cfg_value


def evaluate_scenario(benchmark, game_type, metric, approx, precis, instance=None, param_set=None):
    hpoBenchmark = YahpoGymBenchmark(benchmark, metric)

    if game_type == ABLATION_GAME:
        opt_cfg, _ = get_opt_cfg(hpoBenchmark, instance, random_state=1337)
        game = AblationHPIGame(hpoBenchmark, instance=instance, optimized_cfg=opt_cfg)
    elif game_type == SET_ABLATION_GAME:
        opt_cfg_list = list()
        for i in range(hpoBenchmark.get_num_instances()):
            opt_cfg, _ = get_opt_cfg(hpoBenchmark, i, random_state=1337)
            opt_cfg_list.append(opt_cfg)
        game = AblationSetHPIGame(hpoBenchmark, opt_cfg_list)
    elif game_type == DS_TUNABILITY_GAME:
        game = DataSpecificTunabilityHPIGame(hpoBenchmark=hpoBenchmark, instance=instance)
    elif game_type == TUNABILITY_GAME:
        game = TunabilityHPIGame(hpoBenchmark=hpoBenchmark)
    elif game_type == DS_OPTIMIZER_BIAS_GAME:
        ensemble = [SubspaceRandomOptimizer(random_state=i,
                                            param_selection=hpoBenchmark.get_list_of_tunable_hyperparameters()) for i in range(3)]
        if param_set is None:
            optimizer = LocalOptimizer(random_state=42)
        else:
            optimizer = SubspaceRandomOptimizer(random_state=42, param_selection=param_set)
        game = DataSpecificOptimizerBiasGame(hpoBenchmark=hpoBenchmark, instance=instance, ensemble=ensemble, optimizer=optimizer)
    elif game_type == OPTIMIZER_BIAS_GAME:
        ensemble = [SubspaceRandomOptimizer(random_state=i,
                                            param_selection=hpoBenchmark.get_list_of_tunable_hyperparameters()) for i in range(3)]
        if param_set is None:
            optimizer = LocalOptimizer(random_state=42)
        else:
            optimizer = SubspaceRandomOptimizer(random_state=42, param_selection=param_set)
        game = OptimizerBiasGame(hpoBenchmark=hpoBenchmark, ensemble=ensemble, optimizer=optimizer)
    else:
        print("Requested game not implemented")
        return

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
            param_list += [game.hpoBenchmark.get_list_of_tunable_hyperparameters()[k[i]]]
        value_list += [("+".join(param_list), res.values[v])]
    value_list = sorted(value_list, key=lambda x: x[1])
    for x in value_list:
        print(x)

    matplotlib.use("TkAgg")

    plot = network_plot(
        first_order_values=res.get_n_order_values(1),
        second_order_values=res.get_n_order_values(2),
        feature_names=game.hpoBenchmark.get_list_of_tunable_hyperparameters(),
    )
    plot[0].savefig(
        "hp_importance_" + "_".join([benchmark, game_type, metric, approx, str(precis)]) + ".png"
    )
    print(
        "stored to file ",
        "hp_importance_" + "_".join([benchmark, game_type, metric, approx, str(precis)]) + ".png",
    )


if __name__ == "__main__":
    game_types = [ABLATION_GAME]
    benchmark_list = ["lcbench"]
    metrics = ["val_accuracy"]  # , "bac", "auc", "brier", "f1", "logloss"]
    approx = ["exact"]
    precis_list = [10]

    param_se_a = ["learning_rate", "max_dropout", "max_units"]
    param_set_b = ["learning_rate", "max_dropout", "max_units", "num_layers", "momentum", "batch_size"]
    # benchmark_list = ["lcbench"]
    # metrics = ["val_accuracy"]

    for game_type in game_types:
        for metric in metrics:
            for benchmark in benchmark_list:
                for a in approx:
                    for precis in precis_list:
                        print(benchmark, game_type, metric, a, precis)
                        instance = None
                        if game_type in [ABLATION_GAME, DS_TUNABILITY_GAME, DS_OPTIMIZER_BIAS_GAME]:
                            instance = 0
                        evaluate_scenario(benchmark, game_type, metric, a, precis, instance)
