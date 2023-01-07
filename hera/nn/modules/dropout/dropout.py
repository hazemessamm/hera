import jax

from hera.nn.modules import functional as F
from hera.nn.modules.module import Module
from jax import numpy as jnp


class Dropout(Module):
    def __init__(self, rate: float, rng: int, jit: bool = False):
        """Dropout Module

        Args:
            rate (float): Dropout probability between zero and one.
            rng (int): Initial seed that will be used to create another
                       random seeds each dropout call.
            jit (bool, optional): Whether to JIT compile the forward
                                  method or not. Defaults to False.
        """

        # Stochastic module is set to True only in the case of
        # requiring different random number every time we call it.
        super().__init__(rng, stochastic_module=True, jit=jit)
        self.rate = rate

    def compute_output_shape(self, input_shape):
        inputs = jax.core.ShapedArray((1, *input_shape[1:]), dtype=jnp.float32)
        shape = jax.eval_shape(self.forward, self.parameters(), inputs).shape
        return (None, *shape[1:])

    def pre_forward_hook(self, *args, **kwargs):
        random_key = self.make_random_key()
        return (random_key,)

    def forward(self, weights, inputs, rng):
        out = F.dropout(inputs, weights, self.rate, rng, self.training)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(rate={self.rate})"
