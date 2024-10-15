from downstream_hpo import RSSimulation
from hpo_benchmarks import HyperparameterOptimizationBenchmark


class AbstractOptimizer(ABC):
    def __init__(self, random_state=None, verbose=False):
        self.random_state = random_state
        self.verbose = verbose

    @abstractmethod
    def optimizer(self, hpoBenchmark:HyperparameterOptimizationBenchmark, coalition):
        pass


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


class LocalOptimizer(AbstractOptimizer):
    def __init__(self, budget_per_param=50, random_state=None, verbose=False):
        super().__init__(random_state, verbose)
        self.budget_per_param = budget_per_param

    def optimize(self, hpoBenchmark: HyperparameterOptimizationBenchmark, coalition):
        final_config = dict()
        param_set = hpoBenchmark.get_list_of_tunable_hyperparameters()
        param_sel = list()
        for i, incl in enumerate(coalition):
            if incl == 1:
                param = param_set[i]
                idx_cand, idx_res = RSSimulation(hpoBenchmark, [param], self.budget_per_param).simulate_hpo_run(0)
                final_config[param] = idx_cand[param]

        def_cfg = hpoBenchmark.get_default_config()
        for param in param_sel:
            def_cfg[param] = final_config[param]

        return hpoBenchmark.evaluate(def_cfg)
