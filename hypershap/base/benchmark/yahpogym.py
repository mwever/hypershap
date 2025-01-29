import time
from abc import ABC, abstractmethod

import ConfigSpace
import numpy as np
from ConfigSpace import Configuration, __all__

from hypershap.base.util.dir import YAHPOGYM_FOLDER
from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark

SKIP_PARAMS = ["OpenML_task_id", "task_id"]


class YahpoGymBenchmark(HyperparameterOptimizationBenchmark):
    minimize_metrics = ["time", "runtime_train", "runtime_test", "memory_train", "memory_test"]

    def __init__(self, scenario_name, metric, instance_idx=None, yahpogym_folder=YAHPOGYM_FOLDER):
        from yahpo_gym import benchmark_set, local_config
        local_config.set_data_path(yahpogym_folder)
        self.instance_idx = None

        super().__init__("yahpogym", scenario_name, instance_idx, metric)

        self.benchmark = benchmark_set.BenchmarkSet(scenario_name)
        self.set_instance(instance_idx)

        self.tunable_hyperparameter_names = list()  # list of tunable hyperparameters
        self.non_tunable_hyperparameter_names = list()  # list of non_tunable hyperparameters

        # cache default values for hyperparameters
        self.defaults = dict()

        for hyperparam in self.benchmark.get_opt_space(drop_fidelity_params=True).get_hyperparameters():
            if hyperparam.name not in SKIP_PARAMS:
                self.tunable_hyperparameter_names += [hyperparam.name]
        for hyperparam in self.benchmark.get_opt_space(drop_fidelity_params=False).get_hyperparameters():
            if hyperparam.name not in self.tunable_hyperparameter_names and hyperparam.name not in SKIP_PARAMS:
                self.non_tunable_hyperparameter_names += [hyperparam.name]
        for hp in self.tunable_hyperparameter_names + self.non_tunable_hyperparameter_names:
            hp_obj = self.benchmark.get_opt_space().get_hyperparameter(hp)
            self.defaults[hp] = hp_obj.default_value

    def get_instances(self):
        if self.instance_idx is None:
            return [i for i in range(self.get_num_instances())]
        elif type(self.instance_idx) == list:
            return self.instance_idx
        else:
            return [self.instance_idx]

    def set_instance(self, instance):
        self.instance_idx = instance
        if type(instance) == int:
            self.benchmark.set_instance(self.benchmark.instances[instance])
        elif type(instance) == str:
            self.benchmark.set_instance(instance)

    def get_list_of_tunable_hyperparameters(self):
        return self.tunable_hyperparameter_names

    def get_list_of_nontunable_hyperparameters(self):
        return self.non_tunable_hyperparameter_names

    def evaluate(self, configuration, instance=None, metric=None, maximize=True):
        """
        Evaluate the objective function for the given configuration. This can be done for single instances or across a
        list of instances. In case no instance is given, by default the preset of the benchmark will be used, which if
        kept at its default will iterate over all instances.
        """
        # if no instance is given at all, we will by default evaluate across all instances in the benchmark set
        if instance is None and self.instance_idx is None:
            return [self.evaluate(configuration, instance) for instance in self.benchmark.instances]
        # if the object's instance index is a list, iterate over that list of instances
        elif instance is None and type(self.instance_idx) is list:
            return [self.evaluate(configuration, inst) for inst in self.instance_idx]
        # if the provided instance parameter is a list, iterate over the list of given instances
        elif type(instance) is list:
            return [self.evaluate(configuration, inst) for inst in instance]

        # check whether instance is set and set benchmark to this instance
        change_back = False
        if instance is not None and (type(instance) is int or type(instance) is str):
            if type(instance) is int:
                instance = self.benchmark.instances[instance]
            self.benchmark.set_instance(instance)
            change_back = True

        # if given configuration is not a list of configurations, make it a list
        if type(configuration) is dict:
            configuration = [configuration]

        obj = None
        # iterate over the configurations in the list of configurations and return the maximum/minimum objective value
        for config in configuration:
            base_cfg = self.get_default_config()
            config = dict(config)
            for k in base_cfg.keys():
                if k not in config:
                    config[k] = base_cfg[k]

            success = False
            while not success and len(config) > 0:
                try:
                    if metric is None:
                        m = self.metric
                    else:
                        m = metric

                    res = self.benchmark.objective_function(config)[0][m]
                    if m in self.minimize_metrics:
                        res = (-1) * res

                    if obj is None or res > obj:
                        obj = res
                    success = True
                except ValueError as e:
                    string_error = repr(e)
                    param_to_del = string_error.split("hyperparameter '")[1].split("' must")[0]
                    config.pop(param_to_del, None)
                    print(e)

        # restore instance index originally set
        if change_back and (type(self.instance_idx) is int or type(self.instance_idx) is str):
            self.set_instance(self.instance_idx)
        return obj

    def get_num_instances(self):
        if self.instance_idx is None:
            return len(self.benchmark.instances)
        elif type(self.instance_idx) == list:
            return len(self.instance_idx)
        else:
            return 1

    def get_available_metrics(self):
        return self.benchmark.config.y_names

    def sample_configurations(self, n=1, random_state=None):
        """
        Returns a randomly sampled list of n configurations sampld from the configuration space.
        """
        cfgs = self.benchmark.get_opt_space(drop_fidelity_params=False, seed=random_state).sample_configuration(n)
        if n == 1:
            return [cfgs.get_dictionary()]
        else:
            return [x.get_dictionary() for x in cfgs]

    def get_opt_space(self):
        return self.benchmark.get_opt_space()

    def get_default_config(self):
        def_cfg = self.benchmark.get_opt_space().get_default_configuration().get_dictionary()
        return def_cfg

    def get_default_config_performance(self, instance=None):
        if instance is not None:
            return self.evaluate(self.get_default_config(), instance)
        else:
            return self.evaluate(self.get_default_config())

