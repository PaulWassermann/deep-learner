import deep_learner.nn.optimizer.optimizer as o


class SGD(o.Optimizer):
    def step(self):
        for parameter in self.module.parameters():
            parameter.data = parameter.data - self.learning_rate * parameter.grad.data
