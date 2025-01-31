import json
import math
import os

import shapiq

from hypershap.base.benchmark.yahpogym import YahpoGymBenchmark
from hypershap.downstream_hpo.downstream_hpo import RSSimulation
from hypershap.plots.utils import PARAMETER_NAMES

fanova_hpi_results_folder = "res/fANOVA/"

fanova_dict = {}
fanova_hpis = []
for f in os.listdir(fanova_hpi_results_folder):
    hpi = json.load(open(fanova_hpi_results_folder + f))
    fanova_hpis.append((f, hpi))
    fanova_dict[f.split("_")[4]] = hpi

hypershap_hpi_results_folder = "res/games/yahpogym/"
hypershap_hpis = []
player_names = PARAMETER_NAMES["lcbench"]
outdir = "res/hpo_runs/fanova/"

hypershap_dict = {}
for f in os.listdir(hypershap_hpi_results_folder):
    if "data_specific_tunability" in f and "lcbench" in f:
        hpo_game = shapiq.Game(path_to_values=hypershap_hpi_results_folder + f, normalize=True)
        comp = shapiq.ExactComputer(n_players=hpo_game.n_players, game=hpo_game)
        fsii = comp(index="FSII", order=1)

        instance = f.split("_")[5]

        hpi = dict()
        for i, interaction in enumerate(fsii.get_top_k_interactions(7)):
            if i == 7:
                break
            hpi[player_names[i]] = interaction
        hypershap_dict[instance] = hpi


sensitivity_hpi_results_folder = "res/games/yahpogym-sense/"
sensitivity_dict = {}
for f in os.listdir(sensitivity_hpi_results_folder):
    if "data_specific_tunability" in f and "lcbench" in f:
        hpo_game = shapiq.Game(path_to_values=sensitivity_hpi_results_folder + f, normalize=True)
        comp = shapiq.ExactComputer(n_players=hpo_game.n_players, game=hpo_game)
        fsii = comp(index="FSII", order=1)

        instance = f.split("_")[5]

        hpi = dict()
        for i, interaction in enumerate(fsii.get_top_k_interactions(7)):
            if i == 7:
                break
            hpi[player_names[i]] = interaction
        sensitivity_dict[instance] = hpi

for key in hypershap_dict.keys():
    if key not in fanova_dict.keys():
        continue
    if key not in sensitivity_dict:
        continue

    k = 2
    hs = sorted(list(hypershap_dict[key].items()), key=lambda x: -x[1])[0:k]
    fn = sorted(list(fanova_dict[key].items()), key=lambda x: -x[1])[0:k]
    sense = sorted(list(sensitivity_dict[key].items()), key=lambda x: -x[1])[0:k]

    hs_params = [x[0] for x in hs]
    fn_params = [x[0] for x in fn]
    sense_params = [x[0] for x in sense]

    intersect = set(hs_params).intersection(set(fn_params))
    if len(intersect) < 2:
        print(key)
        print(hs_params, fn_params)
        print(" ")
        hpo_budget = 200
        hpo_runs = 500

        yahpo = YahpoGymBenchmark(scenario_name="lcbench", metric="val_accuracy", instance_idx=key)

        json.dump(top_2_data, open(outdir + key + "_sense_param_names.json", "w"))
        print("Simulate HPO for HyperSHAP")
        # bo_hs = BOSimulation(hpoBenchmark=yahpo, parameter_selection=hs_params, hpo_budget=hpo_budget)
        bo_hs = RSSimulation(
            hpoBenchmark=yahpo, parameter_selection=hs_params, hpo_budget=hpo_budget
        )
        hs_res, hs_std = bo_hs.simulate_hpo(hpo_runs)
        hs_std = hs_std / math.sqrt(hpo_runs)

        print("Simulate HPO for fANOVA")
        bo_fn = RSSimulation(
            hpoBenchmark=yahpo, parameter_selection=fn_params, hpo_budget=hpo_budget
        )
        fn_res, fn_std = bo_fn.simulate_hpo(hpo_runs)
        fn_std = fn_std / math.sqrt(hpo_runs)


        print("Simulate HPO for Sensitivity")
        bo_sense = RSSimulation(
            hpoBenchmark=yahpo, parameter_selection=sense_params, hpo_budget=hpo_budget
        )
        sense_res, sense_std = bo_sense.simulate_hpo(hpo_runs)
        sense_std = sense_std / math.sqrt(hpo_runs)

        print("Simulate HPO for full parameter set")
        fp = RSSimulation(
            hpoBenchmark=yahpo,
            parameter_selection=yahpo.get_list_of_tunable_hyperparameters(),
            hpo_budget=hpo_budget,
        )
        fp_res, fp_std = fp.simulate_hpo(hpo_runs)
        fp_std = fp_std / math.sqrt(hpo_runs)

        print("Determine optimum performance")
        opt = RSSimulation(
            hpoBenchmark=yahpo,
            parameter_selection=yahpo.get_list_of_tunable_hyperparameters(),
            hpo_budget=100_000,
        )
        opt_res, opt_hs_std = opt.simulate_hpo(1)

        # ensure that the maximum is really max of everything seen here
        full_opt = max(fn_res[-1], opt_res[-1], hs_res[-1])

        data_storage = {
            "instance": key,
            "metric": "val_accuracy",
            "hpo_budget": hpo_budget,
            "num_runs": hpo_runs,
            "param_set_hs": hs_params,
            "param_set_sense": sense_params,
            "param_set_fn": fn_params,
            "bo": False,
            "fn_res": fn_res.tolist(),
            "fn_std": fn_std.tolist(),
            "sense_res": sense_res.tolist(),
            "sense_std": sense_std.tolist(),
            "hs_res": hs_res.tolist(),
            "hs_std": hs_std.tolist(),
            "fp_res": fp_res.tolist(),
            "fp_std": fp_std.tolist(),
            "full_opt": str(full_opt),
        }

        with open(outdir + "wref_fanova_vs_hypershap_vs_sensitivity_downstream_" + key + ".json", "w") as outfile:
            json.dump(data_storage, outfile)
