from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark
from hypershap.downstream_hpo.downstream_hpo import RSSimulation


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
