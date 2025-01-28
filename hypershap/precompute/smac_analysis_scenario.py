"""This module contains the surrogate explanation for SMAC."""

import copy
import os
import time

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from sklearn.ensemble import RandomForestRegressor
from smac import HyperparameterOptimizationFacade, Scenario
from smac.facade import AbstractFacade
from smac.initial_design import RandomInitialDesign
from smac.utils.configspace import convert_configurations_to_array

from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark
from hypershap.base.games.tunability import DataSpecificTunabilityHPIGame
from hypershap.downstream_hpo.downstream_hpo import HPOSimulation, LoggingEval


class SMACAnalysisBenchmark(HyperparameterOptimizationBenchmark):

    def __init__(
        self,
        smac: AbstractFacade,
        scenario: Scenario,
        model=None,
        default_performance: float | None = None,
    ) -> None:
        self.smac = smac
        self.scenario = scenario
        self.model = model
        if self.model is None:
            print("Running with default SMAC model")
        self.default_performance = default_performance

    def get_list_of_tunable_hyperparameters(self):
        return [cfg for cfg in self.scenario.configspace.get_hyperparameter_names()]

    def get_list_of_nontunable_hyperparameters(self):
        return []

    def set_instance(self, instance):
        print("WARNING: instance information is ignored in SMACAnalysisBenchmark")

    def get_num_instances(self):
        return 1

    def sample_configurations(self, n=1, random_state=None):
        cfgs = self.scenario.configspace.sample_configuration(n)
        if n == 1:
            return [cfgs.get_dictionary()]
        else:
            return [x.get_dictionary() for x in cfgs]

    def get_opt_space(self):
        return self.scenario.configspace

    def evaluate(self, configuration, instance=None):
        if isinstance(configuration, dict):
            configuration = [configuration]

        cfg_list = [
            Configuration(configuration_space=self.scenario.configspace, values=cfg)
            for cfg in configuration
        ]

        if self.model is None:
            array = convert_configurations_to_array(cfg_list)
            pred = (-1) * self.smac._model.predict(array)
        else:
            array = self._transform_to_array(cfg_list)
            pred = self.model.predict(array)
        return max(pred)

    def get_default_config(self):
        return self.scenario.configspace.get_default_configuration().get_dictionary()

    def get_default_config_performance(self, instance=None):
        if self.default_performance is not None:
            return self.default_performance
        return self.evaluate(self.get_default_config())

    def get_instances(self):
        return []

    @staticmethod
    def _transform_to_array(cfg_list: list[Configuration]) -> np.ndarray:
        result: list[dict] = []
        for config in cfg_list:
            result.append(config._values)
        result_df = pd.DataFrame(result)
        array = result_df.values
        array = array.astype(np.float64)
        return array


class SMACExplanation(HPOSimulation):

    def __init__(
        self,
        run_history_path: str | None,
        hpo_benchmark: HyperparameterOptimizationBenchmark,
        parameter_selection: list,
        hpo_budget: int = 6050,
        random_state: int | None = 0,
    ) -> None:

        super().__init__(hpo_benchmark, parameter_selection, hpo_budget, verbose=False)

        save_dir = "smac_analysis"
        self.file_path = os.path.join(save_dir, "continuous_smac_analysis_lcbench")
        if run_history_path is None:
            run_history_path = os.path.join(save_dir, "smac_lcbench_3945_6050.json")

        # load and parse the run history
        import json

        with open(run_history_path) as f:
            history_data = json.load(f)
        data, target = self.parse_run_history(history_data)
        self.max_budget = len(data)
        self.data = data
        self.target = target

        if self.max_budget != hpo_budget:
            raise ValueError(
                f"Budget {hpo_budget} does not match the number of runs in the run history "
                f"{self.max_budget}"
            )

        self.random_state = random_state
        self.default_config_performance = float(hpo_benchmark.get_default_config_performance())

        # define the surrogate model to mimic SMAC's internal model
        # we use the default hyperparameters for the RandomForestRegressor here

    @staticmethod
    def parse_run_history(history_data: dict) -> (pd.DataFrame, pd.DataFrame):
        """Parses the run History file."""
        runs: list[list] = history_data["data"]
        configs: dict[str, dict] = history_data["configs"]
        result: list[dict] = []
        for run in runs:
            budget: int = run[0]
            performance: float = run[4] * (-1)  # we want to maximize the performance
            config = copy.deepcopy(configs[str(budget)])
            config["performance"] = performance
            result.append(config)
        data = pd.DataFrame(result)
        target = data.pop("performance")
        return data, target

    def inter_run_hook(self):
        time.sleep(1)

    def explain_hpo_run(self, use_own_model: bool = True) -> None:
        eval_fun = LoggingEval(self.hpoBenchmark, self.parameter_selection)

        scenario = Scenario(
            self.reduced_cfg_space,
            deterministic=True,
            n_trials=self.hpo_budget,
            use_default_config=True,
            seed=self.random_state,
        )
        smac = HyperparameterOptimizationFacade(
            scenario,
            eval_fun.train,
            initial_design=RandomInitialDesign(scenario=scenario, n_configs=20),
        )

        budgets = [0.01, 0.03, 0.04, 0.05, 0.25, 0.5, 0.75, 1]
        for budget in budgets:
            if budget < 0:
                raise ValueError(f"Budget {budget} must be non-negative")
            if budget > self.max_budget:
                raise ValueError(
                    f"Budget {budget} is larger than the maximum budget {self.max_budget}"
                )
            if budget < 1 or budget == 1:
                budget = int(budget * self.max_budget)

            file_path = "_".join((self.file_path, str(budget)))
            if use_own_model:
                smac = None
                x_train, y_train = self.data[:budget], self.target[:budget]
                x_train_array, y_train_array = x_train.values, y_train.values
                model = RandomForestRegressor(random_state=self.random_state)
                model.fit(x_train_array, y_train_array)
                print(f"Fit random forest model to budget {budget}")
            else:
                model = None
                file_path += "_smac_model"

            # initialize the tunability game with the new model
            benchmark = SMACAnalysisBenchmark(
                smac, scenario, model=model, default_performance=self.default_config_performance
            )
            game = DataSpecificTunabilityHPIGame(benchmark, 0, 10_000, 42, verbose=True)
            game.precompute()
            file_path += ".npz"
            game.save_values(file_path)
            print(f"Saved to {file_path}")


if __name__ == "__main__":
    from hypershap.base.benchmark.yahpogym import YahpoGymBenchmark

    yahpo = YahpoGymBenchmark(scenario_name="lcbench", instance_idx=0, metric="val_accuracy")
    parameter_selection = yahpo.get_list_of_tunable_hyperparameters()

    sse = SMACExplanation(
        run_history_path=os.path.join("smac_analysis", "smac_lcbench_3945_6050.json"),
        hpo_benchmark=yahpo,
        parameter_selection=parameter_selection,
        hpo_budget=6050,
        random_state=0,
    )

    sse.explain_hpo_run()
