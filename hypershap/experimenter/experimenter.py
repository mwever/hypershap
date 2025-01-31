import argparse
import os
import time

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor

from hypershap.base.util.constants import ABLATION_GAME, DS_TUNABILITY_GAME, TUNABILITY_GAME
from hypershap.precompute.pre_compute_games import _print_game_info
from hypershap.base.util.utils import setup_game, GAME_STORAGE_DIR, _get_game_name


def get_yahpo_fixed_parameter_combinations(with_datasets: bool = True):
    from yahpo_gym import benchmark_set, local_config
    local_config.init_config()
    local_config.set_data_path("yahpodata")

    jobs = []

    # Add all YAHPO-Gym Evaluations
    for scenario in ["rbv2_ranger", "rbv2_xgboost", "rbv2_svm", "rbv2_glmnet", "lcbench", "nb301", "rbv2_aknn", "rbv2_rpart"]:
        bench = benchmark_set.BenchmarkSet(scenario=scenario)

        if "val_accuracy" in bench.config.y_names:
            metric = "val_accuracy"
        elif "acc" in bench.config.y_names:
            metric = "acc"
        else:
            metric = "unknown"

        if with_datasets:
            # create ablation and ds_tunability jobs
            jobs += [{"scenario": scenario, "dataset": dataset, "metric": metric}
                     for dataset in bench.instances]
        else:
            jobs += [{"scenario": scenario, "dataset": "all", "metric": metric}]

    return jobs


def run_experiment(parameters: dict, result_processor:ResultProcessor, custom_config: dict):
    print(parameters)
    benchmarklib = parameters["benchmarklib"]
    scenario = parameters["scenario"]
    dataset = parameters["dataset"]
    metric = parameters["metric"]
    game_id = parameters["game"]
    n_configs = int(parameters["argmax_n_configs"])
    random_state = int(parameters["seed"])

    setup_time = time.time()
    game, _, _, hpo_problem = setup_game(
        game_id,
        benchmark=benchmarklib,
        scenario=scenario,
        metric=metric,
        pre_compute=True,
        verbose=True,
        instance_index=dataset,
        n_configs=n_configs,
    )
    setup_time = time.time() - setup_time
    _print_game_info(game, setup_time)

    # game path
    game_name = _get_game_name(
        game_id, hpo_problem, n_configs=n_configs, random_state=random_state
    )
    game_path = os.path.join(GAME_STORAGE_DIR, f"{game_name}.npz")

    result_processor.process_results({
        "setup_time": setup_time,
        "normalization_value": game.normalization_value,
        "storage_path": game_path,
        "done": "true"
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='HyperSHAP PyExperimenter',
        description='PyExperimenter setup for HyperSHAP'
    )
    parser.add_argument('-b', '--benchmark')
    parser.add_argument('-g', '--game')
    parser.add_argument('-s', '--setup', action='store_true')

    args = parser.parse_args()
    database_cfg_file = "../../conf/database_credentials.yml"

    if args.benchmark == "jahs":
        if args.game == ABLATION_GAME:
            experimenter = PyExperimenter(experiment_configuration_file_path="conf/jahs-experimentsetup_ablation.yml", database_credential_file_path=database_cfg_file)
        elif args.game == DS_TUNABILITY_GAME:
            experimenter = PyExperimenter(experiment_configuration_file_path="conf/jahs-experimentsetup_ds_tunability.yml", database_credential_file_path=database_cfg_file)
        else:
            print("Benchmark ", args.benchmark, " and game ", args.game, " is not a valid combination.")
            exit(1)

        if args.setup:
            experimenter.fill_table_from_config()
    elif args.benchmark == "pd1":
        if args.game == ABLATION_GAME:
            experimenter = PyExperimenter(experiment_configuration_file_path="conf/pd1-experimentsetup_ablation.yml", database_credential_file_path=database_cfg_file)
        elif args.game == DS_TUNABILITY_GAME:
            experimenter = PyExperimenter(experiment_configuration_file_path="conf/pd1-experimentsetup_ds_tunability.yml", database_credential_file_path=database_cfg_file)
        else:
            print("Benchmark ", args.benchmark, " and game ", args.game, " is not a valid combination.")
            exit(1)

        if args.setup:
            experimenter.fill_table_from_config()
    elif args.benchmark == "yahpogym":
        if args.game == ABLATION_GAME:
            experimenter = PyExperimenter(experiment_configuration_file_path="conf/yahpogym-experimentsetup_ablation.yml", database_credential_file_path=database_cfg_file)
        elif args.game == DS_TUNABILITY_GAME:
            experimenter = PyExperimenter(experiment_configuration_file_path="conf/yahpogym-experimentsetup_ds_tunability.yml", database_credential_file_path=database_cfg_file)
        elif args.game == TUNABILITY_GAME:
            experimenter = PyExperimenter(experiment_configuration_file_path="conf/yahpogym-experimentsetup_tunability.yml", database_credential_file_path=database_cfg_file)
        else:
            print("Benchmark ", args.benchmark, " and game ", args.game, " is not a valid combination.")
            exit(1)

        if args.setup:
            experimenter.fill_table_from_combination(
                parameters={
                    "benchmarklib": [args.benchmark],
                    "game": [args.game],
                    "argmax_n_configs": [10_000],
                    "seed": [42]
                },
                fixed_parameter_combinations=get_yahpo_fixed_parameter_combinations(
                    with_datasets=(args.game != TUNABILITY_GAME)
                )
            )
    else:
        print("Unknown benchmark ", args.benchmark, "given")
        exit(1)

    if not args.setup:
        experimenter.execute(experiment_function=run_experiment, max_experiments=1)
