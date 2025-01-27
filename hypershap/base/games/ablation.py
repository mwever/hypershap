import numpy as np

from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark
from hypershap.base.games.abstract_hpi_game import AbstractHPIGame


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