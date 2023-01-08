import abc


class Optimizer(abc.ABC):
    def __init__(self, lr):
        self.lr = lr
        self.initalized = False
        self.optimizer_state = None

    @abc.abstractmethod
    def initialize(self, params):
        pass

    def update_optimizer_state(self, state):
        self.optimizer_state = state
        if not self.initalized:
            self.initalized = True

    @abc.abstractmethod
    def update_weights(self, gradients, params):
        pass
