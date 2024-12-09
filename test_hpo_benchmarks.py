from hpo_benchmarks import YahpoGymBenchmark, PD1Benchmark, JAHSBenchmark


def test_yahpo_benchmark_with_single_instance():
    bench = YahpoGymBenchmark("lcbench", "validation_acc", 0)
    assert bench.get_num_instances() == 1, ("Number of instances should only be 1.")
    assert len(bench.get_available_metrics()) == 6, ("The number of available metrics seems to deviate")

def test_pd1benchmark():
    bench = PD1Benchmark(PD1Benchmark.valid_benchmark_names[0])
    print(bench.get_opt_space())
    print(bench.get_default_config_performance())

    print(bench.sample_configurations(n=1))

    def test_jahs_benchmark():
        bench = JAHSBenchmark("CIFAR10")
        print(bench.get_opt_space())
        print(bench.get_default_config_performance())

        print(bench.sample_configurations(n=1))