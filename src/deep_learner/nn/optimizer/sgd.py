import numpy as np

from deep_learner.nn.optimizer.optimizer import Optimizer


class SGD(Optimizer):
    def step(self):
        for parameter in self.module.parameters():
            parameter.data = parameter.data - self.learning_rate * parameter.grad.data
