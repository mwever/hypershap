from hpo_benchmarks import YahpoGymBenchmark
from hpo_games import AblationHPIGame, DataSpecificTunabilityHPIGame, TunabilityHPIGame, AblationSetHPIGame, \
    DataSpecificOptimizerBiasGame, OptimizerBiasGame
from hpo_optimizers import SubspaceRandomOptimizer, LocalOptimizer
from shapiq import ExactComputer


def test_ablation_hpi_game():
    bench = YahpoGymBenchmark("rbv2_glmnet", "acc", 0)
    opt_cfg = bench.sample_configurations(1, random_state=1337)[0]
    game = AblationHPIGame(bench, 0, opt_cfg, 42)
    shap = ExactComputer(n_players=game.n_players, game_fun=game)
    res = shap(index="k-SII", order=len(bench.get_list_of_tunable_hyperparameters()))
    assert res is not None, "Result should not be None"

def test_set_ablation_hpi_game():
    bench = YahpoGymBenchmark("rbv2_glmnet", "acc")
    opt_cfg = bench.sample_configurations(1, random_state=1337)[0]
    opt_cfg_list = [opt_cfg] * bench.get_num_instances()

    game = AblationSetHPIGame(bench, optimized_cfg_list= opt_cfg_list)
    shap = ExactComputer(n_players=game.n_players, game_fun=game)
    res = shap(index="k-SII", order=len(bench.get_list_of_tunable_hyperparameters()))
    assert res is not None, "Result should not be None"

def test_data_specific_tunability_hpi_game():
    bench = YahpoGymBenchmark("rbv2_glmnet", "acc", 0)
    game = DataSpecificTunabilityHPIGame(bench, instance=0, n_configs=100)
    shap = ExactComputer(n_players=game.n_players, game_fun=game)
    res = shap(index="k-SII", order=len(bench.get_list_of_tunable_hyperparameters()))
    assert res is not None, "Result should not be None"

def test_tunability_hpi_game():
    bench = YahpoGymBenchmark("rbv2_glmnet", "acc")
    game = TunabilityHPIGame(bench, n_configs=10)
    shap = ExactComputer(n_players=game.n_players, game_fun=game)
    res = shap(index="k-SII", order=len(bench.get_list_of_tunable_hyperparameters()))
    assert res is not None, "Result should not be None"

def test_data_specific_optimizer_bias_hpi_game():
    bench = YahpoGymBenchmark("rbv2_glmnet", "acc", instance_idx=0)
    ensemble = [SubspaceRandomOptimizer(random_state=i, hpo_budget=50,
                                        param_selection=bench.get_list_of_tunable_hyperparameters()) for i in range(3)]
    optimizer = LocalOptimizer(random_state=42, budget_per_param=15)
    game = DataSpecificOptimizerBiasGame(bench, instance=0, ensemble=ensemble, optimizer=optimizer)
    shap = ExactComputer(n_players=game.n_players, game_fun=game)
    res = shap(index="k-SII", order=len(bench.get_list_of_tunable_hyperparameters()))
    assert res is not None, "Result should not be None"
def test_optimizer_bias_hpi_game():
    bench = YahpoGymBenchmark("rbv2_glmnet", "acc")
    ensemble = [SubspaceRandomOptimizer(random_state=i, hpo_budget=50,
                                        param_selection=bench.get_list_of_tunable_hyperparameters()) for i in range(3)]
    optimizer = LocalOptimizer(random_state=42, budget_per_param=15)
    game = OptimizerBiasGame(bench, ensemble=ensemble, optimizer=optimizer)
    shap = ExactComputer(n_players=game.n_players, game_fun=game)
    res = shap(index="k-SII", order=len(bench.get_list_of_tunable_hyperparameters()))
    assert res is not None, "Result should not be None"
