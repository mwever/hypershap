import numpy as np

from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark
from hypershap.base.games.abstract_hpi_game import AbstractHPIGame


class TunabilityHPIGame(AbstractHPIGame):
    def __init__(
        self,
        hpoBenchmark: HyperparameterOptimizationBenchmark,
        aggregator=lambda x: np.mean(np.array(x)),
        n_configs=10_000,
        random_state=None,
        verbose: bool = False,
    ):
        self.n_configs = n_configs
        self.aggregator = aggregator

        super().__init__(hpoBenchmark, random_state, verbose)

        self.cfgs = self.hpoBenchmark.sample_configurations(n_configs, random_state)

    def get_default_config_performance(self) -> float:
        return self.aggregator(self.hpoBenchmark.get_default_config_performance())

    def evaluate_single_coalition(self, coalition: np.ndarray):
        if coalition.sum() == 0:
            return self.get_default_config_performance()

        cfgs = self.blind_parameters_according_to_coalition(self.cfgs, coalition)
        obj_list = self.hpoBenchmark.evaluate(cfgs)
        return self.aggregator(obj_list)


class DataSpecificTunabilityHPIGame(TunabilityHPIGame):
    def __init__(
        self, hpoBenchmark, instance, n_configs=10000, random_state=None, verbose: bool = False
    ):
        self.instance = instance
        # ensure the given instance is set in the hpo benchmark before the super constructor is called
        hpoBenchmark.set_instance(instance)
        assert hpoBenchmark.get_num_instances() == 1, (
            "Number of instances cannot exceed 1 for data-specific " "tunability."
        )

        super().__init__(
            hpoBenchmark, n_configs=n_configs, random_state=random_state, verbose=verbose
        )
