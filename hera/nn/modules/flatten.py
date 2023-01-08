import jax
import jax.numpy as jnp
from hera.nn.modules import functional as F
from hera.nn.modules.module import Module


class Flatten(Module):
    def __init__(self, jit=False):
        """Flatten Module.

        Args:
            jit (bool, optional): _description_. Defaults to False.
        """
        super().__init__(jit=jit)

    def compute_output_shape(self, input_shape):
        inputs = jax.core.ShapedArray((1, *input_shape[1:]), dtype=jnp.float32)
        shape = jax.eval_shape(self.forward, self.parameters(), inputs).shape
        return (None, *shape[1:])

    def forward(self, weights, inputs):
        """Flattens the inputs.

        Args:
            weights (Dict): Dictionary with attribute
                            names as keys and weights as values
            inputs (ndarray): n+1-D Tensor with order axis (batch_size, *)

        Returns:
            ndarray: Flattened Tensor
        """
        return F.flatten(weights, inputs)
