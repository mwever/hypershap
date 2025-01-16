import copy
from abc import ABC, abstractmethod
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
            normalization_value=self.get_default_config_performance(),
            verbose=verbose,
            normalize=True,
        )

    def get_default_config_performance(self) -> float:
        return self.hpoBenchmark.get_default_config_performance()

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

    @abstractmethod
    def evaluate_single_coalition(self, coalition):
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
                if key in list_of_hyperparams_to_blind and key in default.keys():
                    cfg[key] = default[key]
        return cfgs


class AblationSetHPIGame(AbstractHPIGame):
    def __init__(
            self,
            hpoBenchmark: HyperparameterOptimizationBenchmark,
            optimized_cfg_list,
            reference_cfg_list=None,
            aggregator=lambda x: np.array(x).mean(),
            random_state=None,
            verbose: bool = False,
    ) -> None:
        self.optimized_cfg_list = optimized_cfg_list
        if reference_cfg_list is not None:
            self.reference_cfg_list = reference_cfg_list
            assert len(self.reference_cfg_list) == len(self.optimized_cfg_list), "Optimized config list must have same length as reference config list"
        else:
            self.reference_cfg_list = [hpoBenchmark.get_default_config()] * len(self.optimized_cfg_list)
        self.aggregator = aggregator
        super().__init__(hpoBenchmark, random_state, verbose)

    def get_default_config_performance(self) -> float:
        return self.aggregator(self.hpoBenchmark.get_default_config_performance())

    def evaluate_single_coalition(self, coalition: np.ndarray):
        cfgs = self.blend_parameters_according_to_coalition(self.optimized_cfg_list, coalition)
        obj = list()
        for i in range(self.hpoBenchmark.get_num_instances()):
            obj += [self.hpoBenchmark.evaluate(cfgs, instance=i)]
        agg_value = self.aggregator(obj)
        return agg_value

    def blend_parameters_according_to_coalition(self, cfgs, coalition):
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

        reference = copy.deepcopy(self.reference_cfg_list)
        default = self.hpoBenchmark.get_default_config()

        for i, cfg in enumerate(cfgs):
            for key in cfg.keys():
                if key in list_of_hyperparams_to_blind and key in default.keys():
                    cfg[key] = reference[i][key]
        return cfgs

class AblationHPIGame(AblationSetHPIGame):
    def __init__(self, hpoBenchmark: HyperparameterOptimizationBenchmark, instance, optimized_cfg, reference_cfg=None, random_state=None,
                 verbose: bool = False):
        self.optimized_cfg = optimized_cfg
        self.reference_cfg = reference_cfg
        hpoBenchmark.set_instance(instance)
        assert hpoBenchmark.get_num_instances() == 1, "Number of instances cannot exceed 1 for ablations."

        reference_cfg_list = [reference_cfg] if reference_cfg is not None else None
        super().__init__(hpoBenchmark, optimized_cfg_list=[optimized_cfg], reference_cfg_list=reference_cfg_list, random_state=random_state, verbose=verbose)

        print(self.optimized_cfg_list)
        print(self.reference_cfg_list)


class TunabilityHPIGame(AbstractHPIGame):
    def __init__(
            self,
            hpoBenchmark: HyperparameterOptimizationBenchmark,
            aggregator=lambda x: np.mean(np.array(x)),
            n_configs=10_000,
            random_state=None,
            verbose: bool = False
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
    def __init__(self, hpoBenchmark, instance, n_configs=10000, random_state=None, verbose: bool = False):
        self.instance = instance
        # ensure the given instance is set in the hpo benchmark before the super constructor is called
        hpoBenchmark.set_instance(instance)
        assert hpoBenchmark.get_num_instances() == 1, ("Number of instances cannot exceed 1 for data-specific "
                                                            "tunability.")

        super().__init__(hpoBenchmark, n_configs=n_configs, random_state=random_state, verbose=verbose)


class OptimizerBiasGame(AbstractHPIGame):
    def __init__(
            self, hpoBenchmark: HyperparameterOptimizationBenchmark, ensemble, optimizer: AbstractOptimizer,
            aggregator=lambda x: np.array(x).mean(), random_state=None, verbose: bool = False):
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
            self, hpoBenchmark: HyperparameterOptimizationBenchmark, instance, ensemble, optimizer, random_state=None,
            verbose: bool = False
    ):
        self.instance = instance
        hpoBenchmark.set_instance(instance)
        assert hpoBenchmark.get_num_instances() == 1, ("Number of instances cannot exceed 1 for data-specific "
                                                            "optimizer bias.")
        super().__init__(hpoBenchmark, ensemble, optimizer, lambda x: np.array(x).mean(), random_state, verbose)


