import logging
import time

import numpy as np
from ConfigSpace import ConfigurationSpace
from tqdm import tqdm

from hypershap.base.hpo_benchmarks import HyperparameterOptimizationBenchmark


def compute_avg_anytime_performance_lines(traces):
    max_length = None
    for t in traces:
        if max_length is None or len(t) > max_length:
            max_length = len(t)

    best_performance_profiles = list()
    for eval_trace in traces:
        best_value = None
        max_profile = list()
        for val in eval_trace:
            if best_value is None or val > best_value:
                best_value = val
            max_profile.append(best_value)

        while len(max_profile) < max_length:
            max_profile.append(best_value)

        best_performance_profiles.append(max_profile)

    best_perf_matrix = np.array(best_performance_profiles)
    avg_best_perf_list = list()
    std_best_perf_list = list()
    for i in range(best_perf_matrix.shape[1]):
        avg_best_perf_list.append(best_perf_matrix[:, i].mean())
        std_best_perf_list.append(best_perf_matrix[:, i].std())
    return np.array(avg_best_perf_list), np.array(std_best_perf_list)


class LoggingEval:

    def __init__(self, hpoBenchmark:HyperparameterOptimizationBenchmark, parameter_selection):
        self.hpoBenchmark = hpoBenchmark
        self.parameter_selection = parameter_selection
        self.trace = list()

    def train(self, config, seed: int = 0):
        orig_config = self.hpoBenchmark.get_default_config()
        req_config = config
        try:
            for param_name in self.parameter_selection:
                orig_config[param_name] = req_config[param_name]
        except KeyError as e:
            print(orig_config)
            print(req_config)
            raise e
        obj = self.hpoBenchmark.evaluate(orig_config)
        self.trace.append(obj)
        return (-1) * obj


class HPOSimulation:

    def __init__(
        self, hpoBenchmark:HyperparameterOptimizationBenchmark, parameter_selection, hpo_budget, verbose=False
    ):
        self.hpoBenchmark = hpoBenchmark
        self.parameter_selection = parameter_selection
        self.hpo_budget = hpo_budget
        self.verbose = verbose

        self.original_cfg_space = self.hpoBenchmark.get_opt_space()
        self.reduced_cfg_space = ConfigurationSpace()
        for param_name in self.parameter_selection:
            self.reduced_cfg_space.add_hyperparameter(self.original_cfg_space.get_hyperparameter(param_name))
        self.cached_traces = list()
        self.current_trace_ix = 0

    def simulate_hpo_run(self, seed=0):
        pass

    def inter_run_hook(self):
        pass

    def simulate_hpo(self, num_runs=10):
        for i in tqdm(range(num_runs)):
            self.simulate_hpo_run(seed=i)
            self.inter_run_hook()
        return compute_avg_anytime_performance_lines(self.cached_traces)


class BOSimulation(HPOSimulation):

    def __init__(
        self, hpoBenchmark:HyperparameterOptimizationBenchmark, parameter_selection, hpo_budget, verbose=False):
        super().__init__(hpoBenchmark, parameter_selection, hpo_budget, verbose)

    def inter_run_hook(self):
        time.sleep(1)

    def simulate_hpo_run(self, seed=0):
        from smac import HyperparameterOptimizationFacade, Scenario

        self.current_trace_ix = len(self.cached_traces)
        self.cached_traces += [list()]

        eval_fun = LoggingEval(
            self.hpoBenchmark,
            self.parameter_selection
        )

        scenario = Scenario(
            self.reduced_cfg_space,
            deterministic=True,
            n_trials=self.hpo_budget,
            use_default_config=True,
            seed=seed,
        )

        while len(self.cached_traces[self.current_trace_ix]) == 0:
            smac = HyperparameterOptimizationFacade(scenario, eval_fun.train, logging_level=logging.WARN)
            smac.optimize()
            self.cached_traces[self.current_trace_ix] = eval_fun.trace

        if self.verbose:
            print(
                self.current_trace_ix,
                np.array(self.cached_traces[self.current_trace_ix]),
                "Length",
                len(self.cached_traces[self.current_trace_ix]),
            )


class RSSimulation(HPOSimulation):
    def __init__(
        self, hpoBenchmark:HyperparameterOptimizationBenchmark, parameter_selection, hpo_budget
    ):
        super().__init__(hpoBenchmark, parameter_selection, hpo_budget)

    def simulate_hpo_run(self, seed=0):
        self.current_trace_ix = len(self.cached_traces)
        self.cached_traces += [list()]

        eval_fun = LoggingEval(
            self.hpoBenchmark,
            self.parameter_selection,
        )

        incumbent = self.hpoBenchmark.get_default_config()
        incumbent_perf = eval_fun.train(incumbent)

        for i in range(self.hpo_budget - 1):
            cfg = self.reduced_cfg_space.sample_configuration()
            res = (-1) * eval_fun.train(cfg)
            if incumbent_perf is None or res > incumbent_perf:
                incumbent_perf = res
                incumbent = cfg

        self.cached_traces[self.current_trace_ix] = eval_fun.trace
        return incumbent, incumbent_perf
