from hypershap.downstream_hpo.downstream_hpo import RSSimulation
from hypershap.base.hpo_benchmarks import HyperparameterOptimizationBenchmark, YahpoGymBenchmark
from hypershap.base.hpo_games import AblationHPIGame
from hypershap.base.hpo_optimizers import SubspaceRandomOptimizer
from shapiq import ExactComputer


class LPI:

    def __init__(self, hpoBenchmark: HyperparameterOptimizationBenchmark):
        self.hpoBenchmark = hpoBenchmark

    def compute_importances(self, config_to_explain, reference_config):
        param_importances = dict()
        non_fixed_params = list(config_to_explain.keys())

        temp_config = reference_config.copy()
        temp_config_perf = self.hpoBenchmark.evaluate(temp_config)

        while len(non_fixed_params) > 0:
            max_eval_param = None
            max_eval_perf = None
            max_eval_config = None

            for param in non_fixed_params:
                eval_config = temp_config.copy()
                eval_config[param] = config_to_explain[param]
                eval_config_perf = self.hpoBenchmark.evaluate(eval_config)

                if max_eval_perf is None or eval_config_perf > max_eval_perf:
                    max_eval_param = param
                    max_eval_config = eval_config
                    max_eval_perf = eval_config_perf

            param_importances[max_eval_param] = max_eval_perf - temp_config_perf
            temp_config = max_eval_config
            temp_config_perf = max_eval_perf
            non_fixed_params.remove(max_eval_param)

        return param_importances


if __name__ == '__main__':
    instance_idx = 0

    hpoBenchmark = YahpoGymBenchmark(scenario_name="lcbench", metric="val_accuracy", instance_idx=instance_idx)
    lpi = LPI(hpoBenchmark)
    rand_opt = SubspaceRandomOptimizer(hpoBenchmark.get_list_of_tunable_hyperparameters(), 10_000)

    rs = RSSimulation(hpoBenchmark, hpoBenchmark.get_list_of_tunable_hyperparameters(), 10_000)

    print("reference config", hpoBenchmark.get_default_config_performance())
    conf, perf = rs.simulate_hpo_run()
    print("opt config", perf)
    importances = lpi.compute_importances(conf, hpoBenchmark.get_default_config())
    print(importances)

    ablationSHAP = AblationHPIGame(hpoBenchmark, instance_idx, dict(conf))
    ec = ExactComputer(ablationSHAP.get_n_players(), ablationSHAP)
    res = ec(index="k-SII", order=ablationSHAP.get_n_players())
    print(res)

    ablationSHAP2 = AblationHPIGame(hpoBenchmark, instance_idx, dict(conf), dict(conf))
    ec = ExactComputer(ablationSHAP2.get_n_players(), ablationSHAP2)
    res = ec(index="k-SII", order=ablationSHAP2.get_n_players())
    print("Same", res)


