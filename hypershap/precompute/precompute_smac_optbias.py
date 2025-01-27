import json
from copy import copy

from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark
from hypershap.base.benchmark.yahpogym import YahpoGymBenchmark
from hypershap.base.optimizer.smac_optimizer import SMACOptimizer
from hypershap.base.util.dir import SMAC_OPTBIAS_GAMES_FOLDER

from joblib import Parallel, delayed


def precompute_smac_optbias(scenario, metric, dataset, coalition, hpo_budget=5000, coal_id=-1):
    hpo_benchmark = YahpoGymBenchmark(scenario_name=scenario, metric=metric, instance_idx=dataset)
    smac_opt = SMACOptimizer(hpo_budget=hpo_budget)
    res = smac_opt.optimize(hpo_benchmark, coalition)

    if coal_id >= 0:
        dump_string = json.dumps((coalition, res))
        with open(SMAC_OPTBIAS_GAMES_FOLDER + "_".join(["smac_optbias", scenario, metric, str(dataset), str(hpo_budget), str(coal_id)]) + ".json", "w") as file:
            file.write(dump_string)

    return coalition, res

def get_all_coalitions(num_members: int):
    if num_members == 1:
        return [[1], [0]]

    a = all_coals_before = get_all_coalitions(num_members - 1)
    b = copy(all_coals_before)
    coals = list()
    for a_i in a:
        coals.append(a_i + [1])
    for b_i in b:
        coals.append(b_i + [0])
    return coals


if __name__ == '__main__':
    from yahpo_gym import benchmark_set, local_config
    local_config.init_config()

    # instantiate hpo benchmark with yahpo gym benchmark
    scenario = "lcbench"
    metric = "val_accuracy"
    dataset = 0
    hpo_budget = 5000

    yahpo = YahpoGymBenchmark(scenario_name=scenario, metric=metric, instance_idx=dataset)

    # output base path
    output_base_path = SMAC_OPTBIAS_GAMES_FOLDER

    # list all possible coalitions
    list_of_coalitions = get_all_coalitions(yahpo.get_number_of_tunable_hyperparameters())

    # run smac instantiations in parallel to quickly assess the coalition values
    results = Parallel(n_jobs=12)(
        delayed(precompute_smac_optbias)(scenario, metric, dataset, coalition, hpo_budget, i) for i, coalition in enumerate(list_of_coalitions))

    print(results)

    dump_string = json.dumps(results)
    with open(output_base_path + "_".join(["smac_optbias", scenario, metric, str(dataset), str(hpo_budget)]) + ".json", "w") as file:
        file.write(dump_string)
