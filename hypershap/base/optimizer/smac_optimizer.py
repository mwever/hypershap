import logging
import json

from ConfigSpace import ConfigurationSpace

from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark
from hypershap.base.optimizer.abstract_optimizer import AbstractOptimizer
from hypershap.base.util.dir import SMAC_OPTBIAS_GAMES_FOLDER


class BenchmarkEvalWrapper:

    def __init__(self, hpo_benchmark: HyperparameterOptimizationBenchmark, maximize=True):
        self.hpoBenchmark = hpo_benchmark
        self.maximize = maximize

    def train(self, config: dict, seed: int = 0):
        orig_config = self.hpoBenchmark.get_default_config()
        req_config = config
        try:
            for param_name in config.keys():
                orig_config[param_name] = req_config[param_name]
        except KeyError as e:
            print(orig_config)
            print(req_config)
            raise e
        obj = self.hpoBenchmark.evaluate(orig_config)
        if self.maximize:
            obj = (-1) * obj
        return obj


class SMACOptimizer(AbstractOptimizer):

    def __init__(self, hpo_budget=1000, maximize=True, random_state=None, verbose=False):
        super().__init__(random_state, verbose)
        self.hpo_budget = hpo_budget
        self.maximize = maximize

    def optimize(self, hpo_benchmark: HyperparameterOptimizationBenchmark, coalition) -> float:
        from smac import HyperparameterOptimizationFacade, Scenario

        # fetch list of tunable hyperparameters from the benchmark
        list_of_tunable_hps = hpo_benchmark.get_list_of_tunable_hyperparameters()

        # assert that number of tunable hps is equal the maximum number of coalition members
        assert len(list_of_tunable_hps) == len(coalition), "Number of tunable HPs deviates from coalition size"

        # construct sub configuration space
        reduced_cfg_space = ConfigurationSpace()
        for i, incl in enumerate(coalition):
            if incl == 1:
                hp = hpo_benchmark.get_opt_space().get_hyperparameter(list_of_tunable_hps[i])
                reduced_cfg_space.add_hyperparameter(hp)

        # setup scenario with reduced config space, given hpo_budget and using the default config
        scenario = Scenario(
            reduced_cfg_space,
            deterministic=True,
            n_trials=self.hpo_budget,
            use_default_config=True,
            seed=42,
        )

        # wrap hpo_benchmark into class providing required train function for smac
        eval_fun = BenchmarkEvalWrapper(hpo_benchmark, self.maximize)

        # instantiate smac with the above scenario and train function, run smac and obtain the incumbent cost
        smac = HyperparameterOptimizationFacade(scenario, eval_fun.train, logging_level=logging.WARN)
        incumbent = smac.optimize()
        incumbent_cost = smac.validate(incumbent)

        # if we wish to maximize the train function wrapper will multiple the result by -1.
        # here we revert this manipulation of the score that is to be maximized
        if self.maximize:
            incumbent_cost = (-1) * incumbent_cost

        return incumbent_cost

class SMACLookUpOptimizer(AbstractOptimizer):

    def __init__(self, verbose=False):
        super().__init__(random_state=None, verbose=verbose)

    def optimize(self, hpo_benchmark: HyperparameterOptimizationBenchmark, coalition):
        instance_idx = 0
        file_pattern = SMAC_OPTBIAS_GAMES_FOLDER + f"smac_optbias_{hpo_benchmark.scenario}_{hpo_benchmark.metric}_{instance_idx}_5000.json"
        coalition_value_list = json.load(open(file_pattern))
        cache = dict()
        for coalition_value_item in coalition_value_list:
            cache[str(coalition_value_item[0])] = coalition_value_item[1]


        query_key = [1 if x else 0 for x in coalition]
        query_key = str(query_key)

        if query_key not in cache:
            print("WARNING: Cache miss!")
        return cache[query_key]
