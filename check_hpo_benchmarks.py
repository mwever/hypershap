from hpo_benchmarks import YahpoGymBenchmark
from hpo_games import AblationHPIGame

bench = YahpoGymBenchmark("lcbench", "val_accuracy")

instance_bench = YahpoGymBenchmark("lcbench", "val_accuracy", 0)
opt_cfg = bench.sample_configurations(n=1)[0]
game = AblationHPIGame(hpoBenchmark=instance_bench, optimized_cfg=opt_cfg)
