import time

from ConfigSpace import ConfigurationSpace
from smac.initial_design import RandomInitialDesign
from smac.runhistory import TrialInfo, TrialValue

from downstream_hpo import LoggingEval, HPOSimulation
from hpo_benchmarks import SMACAnalysisBenchmark, YahpoGymBenchmark, HyperparameterOptimizationBenchmark
from hpo_games import DataSpecificTunabilityHPIGame


class SMACSurrogateExplanation(HPOSimulation):

    def __init__(
        self, hpoBenchmark:HyperparameterOptimizationBenchmark, parameter_selection, hpo_budget, verbose=False
    ):
        super().__init__(hpoBenchmark, parameter_selection, hpo_budget, verbose)

    def inter_run_hook(self):
        time.sleep(1)

    def explain_hpo_run(self, budget_cutoffs, seed=0):
        from smac import HyperparameterOptimizationFacade, Scenario
        eval_fun = LoggingEval(
            self.hpoBenchmark,
            self.parameter_selection
        )

        scenario = Scenario(
            self.reduced_cfg_space,
            deterministic=True,
            n_trials=self.hpo_budget,
            use_default_config=True,
            seed=seed,
        )
        smac = HyperparameterOptimizationFacade(
            scenario, eval_fun.train,
            initial_design=RandomInitialDesign(scenario=scenario, n_configs=20)
        )

        for i in range(self.hpo_budget):
            trial: TrialInfo = smac.ask()
            res = eval_fun.train(trial.config, trial.seed)
            smac.tell(trial, TrialValue(cost=res))

            if i in budget_cutoffs:
                smacAnalysisBenchmark = SMACAnalysisBenchmark(smac, scenario)
                ds_tunability_game = DataSpecificTunabilityHPIGame(smacAnalysisBenchmark, 0, 10_000, 42)
                print("start precompute explanations after a budget of ", i)
                ds_tunability_game.precompute()
                print("end precompute")
                ds_tunability_game.save_values("continuous_smac_analysis_test_" + self.hpoBenchmark.scenario + "_" + str(
                    self.hpoBenchmark.dataset) + "_" + str(i) + ".npz")
                print("saved to file")


yahpo = YahpoGymBenchmark(scenario_name="lcbench", instance_idx=0, metric="val_accuracy")
parameter_selection = yahpo.get_list_of_tunable_hyperparameters()
print(parameter_selection)
sse = SMACSurrogateExplanation(hpoBenchmark=yahpo, parameter_selection=parameter_selection, hpo_budget=6050)
sse.explain_hpo_run(budget_cutoffs=[25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 3000, 6000])