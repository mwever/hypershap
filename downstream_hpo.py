import logging
import time

import numpy as np
from ConfigSpace import ConfigurationSpace
from tqdm import tqdm


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

    def __init__(self, original_cfg_space, parameter_selection, objective_function, metric):
        self.trace = list()
        self.original_cfg_space = original_cfg_space
        self.parameter_selection = parameter_selection
        self.objective_function = objective_function
        self.metric = metric

    def train(self, config, seed: int = 0):
        orig_config = self.original_cfg_space.get_default_configuration().get_dictionary()
        req_config = config.get_dictionary()
        try:
            for param_name in self.parameter_selection:
                orig_config[param_name] = req_config[param_name]
        except KeyError as e:
            print(orig_config)
            print(req_config)
            raise e
        obj = self.objective_function(orig_config)[0][self.metric]
        self.trace.append(obj)
        return (-1) * obj


class HPOSimulation:

    def __init__(
        self, benchmark, metric, parameter_selection, hpo_budget, config_space: ConfigurationSpace
    ):
        self.benchmark = benchmark
        self.metric = metric
        self.parameter_selection = parameter_selection
        self.hpo_budget = hpo_budget
        self.original_cfg_space = config_space
        self.reduced_cfg_space = ConfigurationSpace()
        for param_name in self.parameter_selection:
            self.reduced_cfg_space.add_hyperparameter(config_space.get_hyperparameter(param_name))
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
        self, benchmark, metric, parameter_selection, hpo_budget, config_space: ConfigurationSpace
    ):
        super().__init__(benchmark, metric, parameter_selection, hpo_budget, config_space)

    def inter_run_hook(self):
        time.sleep(1)

    def simulate_hpo_run(self, seed=0):
        from smac import HyperparameterOptimizationFacade, Scenario

        self.current_trace_ix = len(self.cached_traces)
        self.cached_traces += [list()]

        eval_fun = LoggingEval(
            self.original_cfg_space,
            self.parameter_selection,
            self.benchmark.objective_function,
            self.metric,
        )
        scenario = Scenario(
            self.reduced_cfg_space,
            deterministic=True,
            n_trials=self.hpo_budget,
            use_default_config=True,
            seed=seed,
        )
        while len(self.cached_traces[self.current_trace_ix]) == 0:
            smac = HyperparameterOptimizationFacade(
                scenario, eval_fun.train, logging_level=logging.WARN
            )
            smac.optimize()
            self.cached_traces[self.current_trace_ix] = eval_fun.trace
        print(
            self.current_trace_ix,
            np.array(self.cached_traces[self.current_trace_ix]),
            "Length",
            len(self.cached_traces[self.current_trace_ix]),
        )


class RSSimulation(HPOSimulation):
    def __init__(
        self, benchmark, metric, parameter_selection, hpo_budget, config_space: ConfigurationSpace
    ):
        super().__init__(benchmark, metric, parameter_selection, hpo_budget, config_space)

    def simulate_hpo_run(self, seed=0):
        self.current_trace_ix = len(self.cached_traces)
        self.cached_traces += [list()]

        eval_fun = LoggingEval(
            self.original_cfg_space,
            self.parameter_selection,
            self.benchmark.objective_function,
            self.metric,
        )

        incumbent = self.benchmark.get_opt_space().get_default_configuration()
        incumbent_perf = (-1) * eval_fun.train(incumbent)

        for i in range(self.hpo_budget - 1):
            cfg = self.original_cfg_space.sample_configuration()
            res = (-1) * eval_fun.train(cfg)
            if incumbent_perf is None or res > incumbent_perf:
                incumbent_perf = res
                incumbent = cfg

        self.cached_traces[self.current_trace_ix] = eval_fun.trace

        return incumbent, incumbent_perf
