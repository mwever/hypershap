
from abc import ABC, abstractmethod

class HyperparameterOptimizationBenchmark(ABC):

    @abstractmethod
    def get_list_of_tunable_hyperparameters(self):
        pass

    @abstractmethod
    def get_list_of_nontunable_hyperparameters(self):
        pass

    def get_number_of_tunable_hyperparameters(self):
        return len(self.get_list_of_tunable_hyperparameters())

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


SKIP_PARAMS = ["OpenML_task_id", "task_id"]

class YahpoGymBenchmark(HyperparameterOptimizationBenchmark):

    def __init__(self, benchmark_name, metric, instance_idx = None):
        from yahpo_gym import benchmark_set, local_config
        local_config.init_config()
        local_config.set_data_path("yahpodata")
        self.instance_idx = None

        self.benchmark = benchmark_set.BenchmarkSet(benchmark_name)
        self.set_instance(instance_idx)
        self.metric = metric

        self.tunable_hyperparameter_names = list() # list of tunable hyperparameters
        self.non_tunable_hyperparameter_names = list() # list of non_tunable hyperparameters

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

    def evaluate(self, configuration, instance=None):
        """
        Evaluate the objective function for the given configuration. This can be done for single instances or across a
        list of instances. In case no instance is given, by default the preset of the benchmark will be used, which if
        kept at its default will iterate over all instances.
        """
        # if no instance is given at all, we will by default evaluate across all instances in the benchmark set
        if instance is None and self.instance_idx is None:
            return [self.evaluate(configuration, instance) for instance in self.benchmark.instances]
        # if the provided instance parameter is a list, iterate over the list of given instances
        elif type(instance) == list:
            return [self.evaluate(configuration, inst) for inst in instance]
        # if the object's instance index is a list, iterate over that list of instances
        elif type(self.instance_idx) == list:
            return [self.evaluate(configuration, inst) for inst in instance]

        # check whether instance is set and set benchmark to this instance
        change_back = False
        if instance is not None and (type(instance) == int or type(instance) == str):
            if type(instance) == int:
                instance = self.benchmark.instances[instance]
            self.benchmark.set_instance(instance)
            change_back = True

        # if given configuration is not a list of configurations, make it a list
        if type(configuration) is dict:
            configuration = [configuration]

        obj = 0
        # iterate over the configurations in the list of configurations and find the maximum objective value
        for config in configuration:
            success = False
            while not success and len(config) > 0:
                try:
                    obj = max(self.benchmark.objective_function(config)[0][self.metric], obj)
                    success = True
                except ValueError as e:
                    string_error = repr(e)
                    param_to_del = string_error.split("hyperparameter '")[1].split("' must")[0]
                    config.pop(param_to_del, None)

        # restore instance index originally set
        if change_back and (type(self.instance_idx) == int or type(self.instance_idx) == str):
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
