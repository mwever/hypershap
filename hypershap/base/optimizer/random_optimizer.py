from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark
from hypershap.base.optimizer.abstract_optimizer import AbstractOptimizer
from hypershap.downstream_hpo.downstream_hpo import RSSimulation


class SubspaceRandomOptimizer(AbstractOptimizer):
    def __init__(self, param_selection, hpo_budget=50000, random_state=None, verbose=False):
        super().__init__(random_state, verbose)
        self.param_selection = param_selection
        self.hpo_budget = hpo_budget

    def optimize(self, hpoBenchmark:HyperparameterOptimizationBenchmark, coalition):
        param_set = hpoBenchmark.get_list_of_tunable_hyperparameters()
        param_sel = list()
        for i, incl in enumerate(coalition):
            if incl == 1 and param_set[i] in self.param_selection:
                param_sel += [param_set[i]]

        rssim = RSSimulation(hpoBenchmark, param_sel, self.hpo_budget)
        res_cand, res_perf = rssim.simulate_hpo_run(0)
        return max(hpoBenchmark.get_default_config_performance(), res_perf)



class RandomOptimizer(SubspaceRandomOptimizer):
    def __init__(self, hpo_budget=50000, random_state=None, verbose=False):
        super().__init__(None, hpo_budget, random_state, verbose)

    def optimize(self, hpoBenchmark:HyperparameterOptimizationBenchmark, coalition):
        self.param_selection = hpoBenchmark.get_list_of_tunable_hyperparameters()
        return super().optimize(hpoBenchmark, coalition)
