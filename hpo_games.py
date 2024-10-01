import copy
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from downstream_hpo import RSSimulation
from shapiq import Game


class AbstractHyperparameterImportanceGame(Game, ABC):
    def __init__(
        self, bench, metric, random_state: Optional[int] = None, verbose: bool = False
    ) -> None:
        self.bench = bench
        self.metric = metric
        self.random_state = random_state

        self.tunable_hyperparameter_names = list()
        self.non_tunable_hyperparameter_names = list()

        skip_params = ["OpenML_task_id", "task_id"]

        for hyperparam in bench.get_opt_space(drop_fidelity_params=True).get_hyperparameters():
            if hyperparam.name not in skip_params:
                self.tunable_hyperparameter_names += [hyperparam.name]
        for hyperparam in self.bench.get_opt_space(
            drop_fidelity_params=False
        ).get_hyperparameters():
            if (
                hyperparam.name not in self.tunable_hyperparameter_names
                and hyperparam.name not in skip_params
            ):
                self.non_tunable_hyperparameter_names += [hyperparam.name]

        # cache default values for hyperparameters
        self.defaults = dict()
        for hp in self.tunable_hyperparameter_names + self.non_tunable_hyperparameter_names:
            hp_obj = self.bench.get_opt_space().get_hyperparameter(hp)
            self.defaults[hp] = hp_obj.default_value

        # call hook to do remaining setup before the value function is called for the first time
        self._before_first_value_function_hook()

        # determine empty coalition value for normalization
        norm_value = self.value_function(np.zeros((1, len(self.tunable_hyperparameter_names))))[0]

        super().__init__(
            n_players=len(self.tunable_hyperparameter_names),
            normalization_value=norm_value,
            verbose=verbose,
            normalize=True,
        )

    @abstractmethod
    def _before_first_value_function_hook(self):
        pass

    def get_n_players(self):
        return len(self.tunable_hyperparameter_names)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        value_list = []
        for i in range(len(coalitions)):
            value_list += [self.evaluate_single_coalition(coalitions[i])]
        return np.array(value_list)

    def try_error_active_parameters_objective_eval(self, cfg):
        success = False
        while not success:
            try:
                obj = [self.bench.objective_function(cfg)[0][self.metric]]
                success = True
            except ValueError as e:
                string_error = repr(e)
                param_to_del = string_error.split("hyperparameter '")[1].split("' must")[0]
                cfg.pop(param_to_del, None)
        return obj

    def prepare_configs_for_coalition(self, coalition, cfgs):
        pass

    def blind_parameters_according_to_coalition(self, cfgs, coalition):
        cfgs = copy.deepcopy(cfgs)
        list_of_hyperparams_to_blind = np.array(self.tunable_hyperparameter_names)[
            (1 - coalition).astype(bool)
        ]
        list_of_hyperparams_to_blind = np.concatenate(
            (
                list_of_hyperparams_to_blind,
                np.array(self.non_tunable_hyperparameter_names),
            )
        )

        for cfg in cfgs:
            for key in cfg.keys():
                if key in list_of_hyperparams_to_blind:
                    cfg[key] = self.defaults[key]
        return cfgs


class UniversalHyperparameterImportanceGame(AbstractHyperparameterImportanceGame):
    def __init__(
        self,
        bench,
        metric,
        aggregate_instances="mean",
        n_configs=1000,
        random_state=None,
        verbose: bool = False,
        n_instances: Optional[int] = None,
    ):
        self.n_configs = n_configs
        self.aggregate_instances = aggregate_instances
        self.cfgs = list()
        self.n_instances = n_instances

        super().__init__(bench, metric, random_state, verbose=verbose)

    def _before_first_value_function_hook(self):
        configs = self.bench.get_opt_space(
            drop_fidelity_params=False, seed=self.random_state
        ).sample_configuration(self.n_configs)
        for cfg in configs:
            self.cfgs += [cfg.get_dictionary()]

    def evaluate_single_coalition(self, coalition: np.ndarray):
        cfgs = self.blind_parameters_according_to_coalition(self.cfgs, coalition)
        obj = list()

        for i, instance in enumerate(self.bench.instances):
            if self.n_instances is not None and i >= self.n_instances:
                break
            self.bench.set_instance(instance)
            obj += [
                np.array(
                    [self.try_error_active_parameters_objective_eval(cfg) for cfg in cfgs]
                ).max()
            ]

        if self.aggregate_instances == "median":
            agg_value = np.median(np.array(obj))
        elif self.aggregate_instances == "max":
            agg_value = np.array(obj).max()
        elif self.aggregate_instances == "mean":
            agg_value = np.array(obj).mean()
        elif self.aggregate_instances == "min":
            agg_value = np.array(obj).min()
        return agg_value


class GlobalHyperparameterImportanceGame(AbstractHyperparameterImportanceGame):
    def __init__(self, bench, metric, n_configs=1000, random_state=None, verbose: bool = False):
        self.n_configs = n_configs
        self.cfgs = list()
        super().__init__(bench, metric, random_state, verbose=verbose)

    def _before_first_value_function_hook(self):
        configs = self.bench.get_opt_space(
            drop_fidelity_params=False, seed=self.random_state
        ).sample_configuration(self.n_configs)
        self.cfgs = [cfg.get_dictionary() for cfg in configs]

    def evaluate_single_coalition(self, coalition: np.ndarray):
        cfgs = self.blind_parameters_according_to_coalition(self.cfgs, coalition)
        obj = [self.try_error_active_parameters_objective_eval(cfg) for cfg in cfgs]
        return np.array(obj).max()


class LocalHyperparameterImportanceGame(AbstractHyperparameterImportanceGame):
    def __init__(self, bench, metric, optimized_cfg, verbose: bool = False):
        self.optimized_cfg = optimized_cfg
        super().__init__(bench, metric, verbose=verbose)

    def _before_first_value_function_hook(self):
        pass

    def evaluate_single_coalition(self, coalition: np.ndarray):
        cfg = self.blind_parameters_according_to_coalition([self.optimized_cfg], coalition)[0]
        return self.try_error_active_parameters_objective_eval(cfg)


class UniversalLocalHyperparameterImportanceGame(AbstractHyperparameterImportanceGame):
    def __init__(
        self,
        bench,
        metric,
        optimized_cfg_list,
        aggregate_instances: str = "mean",
        verbose: bool = False,
    ) -> None:
        self.optimized_cfg_list = optimized_cfg_list
        self.aggregate_instances = aggregate_instances
        super().__init__(bench, metric, verbose=verbose)

    def _before_first_value_function_hook(self):
        pass

    def evaluate_single_coalition(self, coalition: np.ndarray):
        cfgs = self.blind_parameters_according_to_coalition(self.optimized_cfg_list, coalition)
        obj = list()

        for i, instance in enumerate(self.bench.instances):
            self.bench.set_instance(instance)
            obj += [self.try_error_active_parameters_objective_eval(cfgs[i])]

        if self.aggregate_instances == "median":
            agg_value = np.median(np.array(obj))
        elif self.aggregate_instances == "max":
            agg_value = np.array(obj).max()
        elif self.aggregate_instances == "mean":
            agg_value = np.array(obj).mean()
        elif self.aggregate_instances == "min":
            agg_value = np.array(obj).min()

        return agg_value


class OptimizerBiasGame(GlobalHyperparameterImportanceGame):

    def __init__(
        self, bench, metric, optimizer, n_configs=10000, random_state=None, verbose: bool = False
    ):
        self.optimizer = optimizer
        super().__init__(bench, metric, n_configs, random_state, verbose)

    def evaluate_single_coalition(self, coalition: np.ndarray):
        opt_res = self.optimizer.optimize(coalition)
        gt_res = max(super().evaluate_single_coalition(coalition), opt_res)
        # print(coalition, (opt_res - gt_res), "gt", gt_res, "opt", opt_res)
        return opt_res - gt_res


class SubspaceRandomOptimizer:
    def __init__(self, bench, metric, random_state, param_selection):
        self.bench = bench
        self.metric = metric
        self.random_state = random_state
        self.param_set = list()
        for hyperparam in bench.get_opt_space().get_hyperparameters():
            if hyperparam.name not in ["OpenML_task_id", "epoch"]:
                self.param_set += [hyperparam.name]
        self.param_selection = param_selection
        self.default_perf = self.bench.objective_function(
            self.bench.get_opt_space().get_default_configuration().get_dictionary()
        )[0][self.metric]

    def optimize(self, coalition):
        param_sel = list()
        for i, incl in enumerate(coalition):
            if incl == 1:
                param_sel += [self.param_set[i]]

        coal_param_sel = list()
        for x in self.param_selection:
            if x in param_sel:
                coal_param_sel += [x]

        rssim = RSSimulation(
            self.bench, self.metric, coal_param_sel, 50000, self.bench.get_opt_space()
        )
        res_cand, res_perf = rssim.simulate_hpo_run(0)
        return max(self.default_perf, res_perf)


class LocalOptimizer:
    def __init__(self, bench, metric, random_state, budget_per_param=50):
        self.bench = bench
        self.metric = metric
        self.random_state = random_state
        self.param_set = list()
        for hyperparam in bench.get_opt_space().get_hyperparameters():
            if hyperparam.name not in ["OpenML_task_id", "epoch"]:
                self.param_set += [hyperparam.name]
        self.budget_per_param = budget_per_param
        self.default_perf = self.bench.objective_function(
            self.bench.get_opt_space().get_default_configuration().get_dictionary()
        )[0][self.metric]

    def optimize(self, coalition):
        final_config = dict()

        param_sel = list()
        for i, incl in enumerate(coalition):
            if incl == 1:
                param_sel += [self.param_set[i]]
        for param in param_sel:
            idx_cand, idx_res = RSSimulation(
                self.bench, self.metric, [param], self.budget_per_param, self.bench.get_opt_space()
            ).simulate_hpo_run(0)
            final_config[param] = idx_cand[param]

        def_cfg = self.bench.get_opt_space().get_default_configuration().get_dictionary()
        for param in param_sel:
            def_cfg[param] = final_config[param]
        return self.bench.objective_function(def_cfg)[0][self.metric]
