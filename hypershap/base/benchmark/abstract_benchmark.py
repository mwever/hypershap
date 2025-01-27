from abc import ABC, abstractmethod


class HyperparameterOptimizationBenchmark(ABC):

    def __init__(self, benchmark_lib, scenario, dataset, metric):
        self.benchmark_lib = benchmark_lib
        self.scenario = scenario
        self.dataset = dataset
        self.metric = metric

    @abstractmethod
    def get_list_of_tunable_hyperparameters(self):
        pass

    @abstractmethod
    def get_list_of_nontunable_hyperparameters(self):
        pass

    def get_number_of_tunable_hyperparameters(self):
        return len(self.get_list_of_tunable_hyperparameters())

    @abstractmethod
    def set_instance(self, instance):
        pass

    @abstractmethod
    def get_num_instances(self):
        pass

    @abstractmethod
    def sample_configurations(self, n=1, random_state=None):
        pass

    @abstractmethod
    def get_opt_space(self):
        pass

    @abstractmethod
    def evaluate(self, configuration, instance=None):
        pass

    @abstractmethod
    def get_default_config(self):
        pass

    @abstractmethod
    def get_default_config_performance(self, instance=None):
        pass

    @abstractmethod
    def get_instances(self):
        pass
