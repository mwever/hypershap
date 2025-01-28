import json
import os
from copy import copy

from hypershap.base.benchmark.yahpogym import YahpoGymBenchmark
from hypershap.base.games.optimizer_bias import DataSpecificOptimizerBiasGame
from hypershap.base.optimizer.random_optimizer import  RandomOptimizer
from hypershap.base.optimizer.smac_optimizer import SMACOptimizer, SMACLookUpOptimizer
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
    only_load_precomputed_results = True

    # instantiate hpo benchmark with yahpo gym benchmark
    scenario = "lcbench"
    metric = "val_accuracy"
    dataset = 0
    hpo_budget = 5000

    yahpo = YahpoGymBenchmark(scenario_name=scenario, metric=metric, instance_idx=dataset)

    # output base path
    output_base_path = SMAC_OPTBIAS_GAMES_FOLDER
    file_name = "_".join(["smac_optbias", scenario, metric, str(dataset), str(hpo_budget)])
    out_file = output_base_path + file_name + ".json"

    if not only_load_precomputed_results:
        # list all possible coalitions
        list_of_coalitions = get_all_coalitions(yahpo.get_number_of_tunable_hyperparameters())
        # run smac instantiations in parallel to quickly assess the coalition values
        results = Parallel(n_jobs=12)(
            delayed(precompute_smac_optbias)(scenario, metric, dataset, coalition, hpo_budget, i) for i, coalition in enumerate(list_of_coalitions))
        dump_string = json.dumps(results)
        with open(out_file, "w") as file:
            file.write(dump_string)

    ensemble = [RandomOptimizer(hpo_budget=50000), RandomOptimizer(hpo_budget=5000)]
    optimizer = SMACLookUpOptimizer()

    game = DataSpecificOptimizerBiasGame(hpoBenchmark=yahpo, instance=dataset, ensemble=ensemble, optimizer=optimizer, verbose=True)

    game.precompute()
    game_path = SMAC_OPTBIAS_GAMES_FOLDER + file_name + ".npz"
    game.save_values(game_path)

    # save the player names
    name_file = os.path.join(SMAC_OPTBIAS_GAMES_FOLDER, f"{yahpo.benchmark_lib}_{yahpo.scenario}_{yahpo.dataset}.names")
    with open(name_file, "w") as f:
        f.write("\n".join(yahpo.get_list_of_tunable_hyperparameters()))
