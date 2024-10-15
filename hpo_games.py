import copy
from abc import ABC
from typing import Optional

import numpy as np

from hpo_optimizers import AbstractOptimizer
from shapiq import Game

from hpo_benchmarks import HyperparameterOptimizationBenchmark


class AbstractHPIGame(Game, ABC):
    """
    Abstract class for hyperparameter importance games.
    """

    def __init__(
            self,
            hpoBenchmark: HyperparameterOptimizationBenchmark,
            random_state: Optional[int] = None,
            verbose: bool = False
    ) -> None:
        self.hpoBenchmark = hpoBenchmark
        self.random_state = random_state
        # determine empty coalition value for normalization
        super().__init__(
            n_players=self.hpoBenchmark.get_number_of_tunable_hyperparameters(),
            normalization_value=self.hpoBenchmark.get_default_config_performance(),
            verbose=verbose,
            normalize=True,
        )

    def _before_first_value_function_hook(self):
        pass

    def get_n_players(self):
        return self.hpoBenchmark.get_number_of_tunable_hyperparameters()

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        value_list = []
        for i in range(len(coalitions)):
            value_list += [self.evaluate_single_coalition(coalitions[i])]
        return np.array(value_list)

    def prepare_configs_for_coalition(self, coalition, cfgs):
        pass

    def blind_parameters_according_to_coalition(self, cfgs, coalition):
        cfgs = copy.deepcopy(cfgs)
        list_of_hyperparams_to_blind = np.array(self.hpoBenchmark.get_list_of_tunable_hyperparameters())[
            (1 - coalition).astype(bool)
        ]
        list_of_hyperparams_to_blind = np.concatenate(
            (
                list_of_hyperparams_to_blind,
                np.array(self.hpoBenchmark.get_list_of_nontunable_hyperparameters()),
            )
        )

        default = self.hpoBenchmark.get_default_config()
        for cfg in cfgs:
            for key in cfg.keys():
                if key in list_of_hyperparams_to_blind:
                    cfg[key] = default[key]
        return cfgs


class AblationSetHPIGame(AbstractHPIGame):
    def __init__(
            self,
            hpoBenchmark: HyperparameterOptimizationBenchmark,
            optimized_cfg_list,
            aggregator=lambda x: np.array(x).mean(),
            random_state=None,
            verbose: bool = False,
    ) -> None:
        super().__init__(hpoBenchmark, random_state, verbose)
        self.optimized_cfg_list = optimized_cfg_list
        self.aggregator = aggregator

    def evaluate_single_coalition(self, coalition: np.ndarray):
        cfgs = self.blind_parameters_according_to_coalition(self.optimized_cfg_list, coalition)
        obj = list()

        for i, instance in enumerate(self.bench.instances):
            self.bench.set_instance(instance)
            obj += [self.try_error_active_parameters_objective_eval(cfgs[i])]

        agg_value = self.aggregator(obj)
        return agg_value


class AblationHPIGame(AblationSetHPIGame):
    def __init__(self, hpoBenchmark: HyperparameterOptimizationBenchmark, optimized_cfg, random_state=None,
                 verbose: bool = False):
        super().__init__(hpoBenchmark, optimized_cfg_list=[optimized_cfg], random_state=random_state, verbose=verbose)
        self.optimized_cfg = optimized_cfg
        assert self.hpoBenchmark.get_num_instances() == 1, "Number of instances cannot exceed 1 for ablations."

    def evaluate_single_coalition(self, coalition: np.ndarray):
        cfg = self.blind_parameters_according_to_coalition([self.optimized_cfg], coalition)[0]
        return self.hpoBenchmark.evaluate(cfg)


class TunabilityHPIGame(AbstractHPIGame):
    def __init__(
            self,
            hpoBenchmark: HyperparameterOptimizationBenchmark,
            aggregator=lambda x: np.mean(np.array(x)),
            n_configs=1000,
            random_state=None,
            verbose: bool = False
    ):
        super().__init__(hpoBenchmark, random_state, verbose)
        self.n_configs = n_configs
        self.aggregator = aggregator
        self.cfgs = self.hpoBenchmark.sample_configurations(n_configs, random_state)

    def evaluate_single_coalition(self, coalition: np.ndarray):
        cfgs = self.blind_parameters_according_to_coalition(self.cfgs, coalition)
        obj_list = self.hpoBenchmark.evaluate(cfgs)
        return self.aggregator(obj_list)


class DataSpecificTunabilityHPIGame(TunabilityHPIGame):
    def __init__(self, hpoBenchmark, instance, metric, n_configs=1000, random_state=None, verbose: bool = False):
        super().__init__(hpoBenchmark, n_configs=n_configs, random_state=random_state, verbose=verbose)
        hpoBenchmark.instance = instance
        assert self.hpoBenchmark.get_num_instances() == 1, ("Number of instances cannot exceed 1 for data-specific "
                                                            "tunability.")


class OptimizerBiasGame(AbstractHPIGame):
    def __init__(
            self, hpoBenchmark: HyperparameterOptimizationBenchmark, ensemble, optimizer: AbstractOptimizer, aggregator,
            random_state=None, verbose: bool = False):
        super().__init__(hpoBenchmark, random_state, verbose)
        self.ensemble = ensemble
        self.optimizer = optimizer
        self.aggregator = aggregator

    def evaluate_single_coalition(self, coalition: np.ndarray):
        opt_res = self.optimizer.optimize(self.hpoBenchmark, coalition)
        ensemble_res = opt_res
        for member in self.ensemble(coalition):
            member_res = member.optimize(self.hpoBenchmark, coalition)
            for i in range(len(member_res)):
                ensemble_res[i] = max(member_res[i], ensemble_res[i])

        return self.aggregator((np.array(opt_res) - np.array(ensemble_res)).tolist())


class DataSpecificOptimizerBiasGame(OptimizerBiasGame):
    def __init__(
            self, hpoBenchmark: HyperparameterOptimizationBenchmark, ensemble, optimizer, random_state=None,
            verbose: bool = False
    ):
        super().__init__(hpoBenchmark, ensemble, optimizer, lambda x: np.array(x).mean(), random_state, verbose)
        assert self.hpoBenchmark.get_num_instances() == 1, ("Number of instances cannot exceed 1 for data-specific "
                                                            "optimizer bias.")
