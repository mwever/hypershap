import copy
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from shapiq import Game

from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark


class AbstractHPIGame(Game, ABC):
    """
    Abstract class for hyperparameter importance games.
    """

    def __init__(
        self,
        hpoBenchmark: HyperparameterOptimizationBenchmark,
        random_state: Optional[int] = None,
        verbose: bool = False,
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
        list_of_hyperparams_to_blind = np.array(
            self.hpoBenchmark.get_list_of_tunable_hyperparameters()
        )[(1 - coalition).astype(bool)]
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
