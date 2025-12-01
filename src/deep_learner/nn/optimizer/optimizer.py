from abc import ABC, abstractmethod

import deep_learner.nn.module as m


class Optimizer(ABC):
    def __init__(self, module: m.Module, learning_rate: float):
        self.learning_rate: float = learning_rate
        self.module: m.Module = module

    @abstractmethod
    def step(self): ...

    def zero_grad(self):
        for parameter in self.module.parameters():
            parameter.zero_grad()
