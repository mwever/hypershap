import numpy as np

from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark
from hypershap.base.games.abstract_hpi_game import AbstractHPIGame
from hypershap.base.optimizer.abstract_optimizer import AbstractOptimizer


class OptimizerBiasGame(AbstractHPIGame):
    def __init__(
        self,
        hpoBenchmark: HyperparameterOptimizationBenchmark,
        ensemble,
        optimizer: AbstractOptimizer,
        aggregator=lambda x: np.array(x).mean(),
        random_state=None,
        verbose: bool = False,
    ):
        self.ensemble = ensemble
        self.optimizer = optimizer
        self.aggregator = aggregator

        super().__init__(hpoBenchmark, random_state, verbose)

    def get_default_config_performance(self) -> float:
        return self.aggregator(self.hpoBenchmark.get_default_config_performance())

    def evaluate_single_coalition(self, coalition: np.ndarray):
        opt_list = list()
        ensemble_list = list()
        instances = self.hpoBenchmark.get_instances()

        for inst in instances:
            self.hpoBenchmark.set_instance(inst)

            opt_res = self.optimizer.optimize(self.hpoBenchmark, coalition)
            ensemble_res = opt_res

            for member in self.ensemble:
                member_res = member.optimize(self.hpoBenchmark, coalition)
                ensemble_res = max(member_res, ensemble_res)

            opt_list.append(opt_res)
            ensemble_list.append(ensemble_res)
        self.hpoBenchmark.set_instance(instances)
        agg = self.aggregator((np.array(opt_list) - np.array(ensemble_list)).tolist())
        return agg


class DataSpecificOptimizerBiasGame(OptimizerBiasGame):
    def __init__(
        self,
        hpoBenchmark: HyperparameterOptimizationBenchmark,
        instance,
        ensemble,
        optimizer,
        random_state=None,
        verbose: bool = False,
    ):
        self.instance = instance
        hpoBenchmark.set_instance(instance)
        assert hpoBenchmark.get_num_instances() == 1, (
            "Number of instances cannot exceed 1 for data-specific " "optimizer bias."
        )
        super().__init__(
            hpoBenchmark, ensemble, optimizer, lambda x: np.array(x).mean(), random_state, verbose
        )
