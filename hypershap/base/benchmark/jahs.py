import ConfigSpace

from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark

class JAHSBenchmark(HyperparameterOptimizationBenchmark):

    valid_dataset_names = ["CIFAR10", "ColorectalHistology", "FashionMNIST"]

    def __init__(self, dataset):
        import mfpbench
        super().__init__("jahs","jahs", dataset, "default")
        self.dataset = dataset
        self.bench = mfpbench.get(name="jahs", task_id=dataset)
        self.eval_cache = {}
        self.tunable_hyperparameters = []
        self.non_tunable_hyperparameters = []

        for hp in self.get_opt_space().get_hyperparameters():
            if not isinstance(hp, ConfigSpace.Constant):
                self.tunable_hyperparameters.append(hp.name)
            else:
                self.non_tunable_hyperparameters.append(hp.name)

    def get_list_of_tunable_hyperparameters(self):
        return self.tunable_hyperparameters

    def get_list_of_nontunable_hyperparameters(self):
        return self.non_tunable_hyperparameters

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
        from mfpbench.jahs import JAHSResult
        if type(configuration) is dict:
            configuration = [configuration]

        obj = None
        # iterate over the configurations in the list of configurations and return the maximum/minimum objective value
        for config in configuration:
            key = hash(frozenset(config.items()))

            success = False
            while not success and len(config) > 0:
                try:
                    if key in self.eval_cache:
                        res = self.eval_cache[key]
                    else:
                        res = self.bench.query(config)
                        if isinstance(res, JAHSResult):
                            jahsres: JAHSResult = res
                            res = jahsres.valid_acc / 100
                            self.eval_cache[key] = res
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
