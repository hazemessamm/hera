import optax
import hera
from hera.nn.optimizers.base_optimizer import Optimizer
import jax
from hera.nn import Module
from typing import Tuple


class Adam(Optimizer):
    def __init__(
        self,
        module: Module,
        lr: float,
        betas: Tuple = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(module=module, lr=lr)
        adam_optimizer = optax.adam(lr, b1=betas[0], b2=betas[1], eps=eps)
        self.init_fn = adam_optimizer.init
        self.update_fn = jax.jit(adam_optimizer.update)

    def initialize(self, params):
        self.update_optimizer_state(self.init_fn(params))

    def step(self, gradients):
        if not self.initalized:
            self.initialize(self.module.parameters())

        updates, opt_state = self.update_fn(gradients, self.optimizer_state)
        self.update_optimizer_state(state=opt_state)
        new_weights = hera.apply_updates(
            params=self.module.parameters(), updates=updates
        )
        self.module.update_parameters(new_weights)
