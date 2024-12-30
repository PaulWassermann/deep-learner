from abc import ABC, abstractmethod

from deep_learner.nn import Module


class Optimizer(ABC):
    def __init__(self, module: Module, learning_rate: float):
        self.learning_rate: float = learning_rate
        self.module: Module = module

    @abstractmethod
    def step(self):
        ...

    def zero_grad(self):
        for parameter in self.module.parameters():
            parameter.zero_grad()
