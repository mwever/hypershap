from abc import ABC, abstractmethod

from hypershap.base.benchmark.abstract_benchmark import HyperparameterOptimizationBenchmark


class AbstractOptimizer(ABC):
    def __init__(self, random_state=None, verbose=False):
        self.random_state = random_state
        self.verbose = verbose

    @abstractmethod
    def optimize(self, hpo_benchmark: HyperparameterOptimizationBenchmark, coalition):
        pass

