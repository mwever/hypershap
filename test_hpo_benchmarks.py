from hpo_benchmarks import YahpoGymBenchmark


def test_yahpo_benchmark_with_single_instance():
    bench = YahpoGymBenchmark("lcbench", "validation_acc", 0)
    assert bench.get_num_instances() == 1, ("Number of instances should only be 1.")
    assert len(bench.get_available_metrics()) == 6, ("The number of available metrics seems to deviate")

