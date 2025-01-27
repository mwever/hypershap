import time
from abc import ABC, abstractmethod

import ConfigSpace
import numpy as np
from ConfigSpace import Configuration, __all__
from smac import Scenario
from smac.facade import AbstractFacade
from smac.utils.configspace import convert_configurations_to_array

from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark


class SMACAnalysisBenchmark(HyperparameterOptimizationBenchmark):

    def __init__(self, smac: AbstractFacade, scenario: Scenario):
        self.smac = smac
        self.scenario = scenario


    def get_list_of_tunable_hyperparameters(self):
        return [cfg for cfg in self.scenario.configspace.get_hyperparameter_names()]

    def get_list_of_nontunable_hyperparameters(self):
        return []

    def set_instance(self, instance):
        print("WARNING: instance information is ignored in SMACAnalysisBenchmark")

    def get_num_instances(self):
        return 1

    def sample_configurations(self, n=1, random_state=None):
        cfgs = self.scenario.configspace.sample_configuration(n)
        if n == 1:
            return [cfgs.get_dictionary()]
        else:
            return [x.get_dictionary() for x in cfgs]

    def get_opt_space(self):
        return self.scenario.configspace

    def evaluate(self, configuration, instance=None):
        if type(configuration) is dict:
            configuration = [configuration]

        cfg_list = [Configuration(configuration_space=self.scenario.configspace, values=cfg) for cfg in configuration]
        array = convert_configurations_to_array(cfg_list)
        print("juhu")
        print(array)
        y_hat = self.smac._model.predict(array)
        print("y hat", y_hat)
        pred = self.smac._model.predict(array)
        print(pred)
        return pred

    def get_default_config(self):
        return self.scenario.configspace.get_default_configuration().get_dictionary()

    def get_default_config_performance(self, instance=None):
        return self.evaluate(self.get_default_config())

    def get_instances(self):
        return []
