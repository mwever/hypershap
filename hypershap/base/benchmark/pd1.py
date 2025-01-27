import time
from abc import ABC, abstractmethod

import ConfigSpace
import numpy as np
from ConfigSpace import Configuration, __all__

from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark

class PD1Benchmark(HyperparameterOptimizationBenchmark):

    valid_benchmark_names = ["lm1b_transformer_2048", "translatewmt_xformer_64", "cifar100_wideresnet_2048", "imagenet_resnet_512"]
    # "uniref50_transformer_128",
    tunable_hyperparameters = []
    non_tunable_hyperparameters = []

    def __init__(self, scenario_name):
        import mfpbench
        super().__init__("pd1", scenario_name, "default", "default")

        if scenario_name not in PD1Benchmark.valid_benchmark_names:
            raise ValueError("Invalid benchmark name")

        self.bench = mfpbench.get(scenario_name)


    def get_list_of_tunable_hyperparameters(self):
        return self.get_opt_space().get_hyperparameter_names()

    def get_list_of_nontunable_hyperparameters(self):
        return []

    def set_instance(self, instance):
        pass

    def get_num_instances(self):
        return 1

    def sample_configurations(self, n: int = 1, random_state=None):
        cfgs = self.bench.sample(n=n, seed=random_state)
        return [x.as_dict() for x in cfgs]

    def get_opt_space(self):
        return self.bench.space

    def evaluate(self, configuration, instance=None):
        if type(configuration) is dict:
            configuration = [configuration]

        from mfpbench.pd1.benchmark import PD1ResultTransformer
        obj = None
        # iterate over the configurations in the list of configurations and return the maximum/minimum objective value
        for config in configuration:
            success = False
            while not success and len(config) > 0:
                try:
                    res = self.bench.query(config)
                    if isinstance(res, PD1ResultTransformer):
                        pd1res: PD1ResultTransformer = res
                        res = pd1res.score
                    else:
                        print(res.__class__)
                        exit()

                    if obj is None or res > obj:
                        obj = res
                    success = True
                except ValueError as e:
                    string_error = repr(e)
                    param_to_del = string_error.split("hyperparameter '")[1].split("' must")[0]
                    config.pop(param_to_del, None)

        # restore instance index originally set
        return obj

    def get_default_config(self):
        return self.bench.space.get_default_configuration().get_dictionary()

    def get_default_config_performance(self, instance=None):
        return self.evaluate(self.get_default_config(), instance)

    def get_instances(self):
        return [ "1" ]
