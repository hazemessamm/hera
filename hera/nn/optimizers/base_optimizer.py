import abc

from hera.nn import Module


class Optimizer(abc.ABC):
    def __init__(self, module: Module, lr: float):
        self.module = module
        self.lr = lr
        self.initialized = False
        self.optimizer_state = None

    @abc.abstractmethod
    def initialize(self, params):
        pass

    def update_optimizer_state(self, state):
        self.optimizer_state = state
        if not self.initialized:
            self.initialized = True

    @abc.abstractmethod
    def step(self, gradients):
        pass
