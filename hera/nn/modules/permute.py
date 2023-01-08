from typing import Tuple

import jax
import jax.numpy as jnp

from hera.nn.modules import functional as F
from hera.nn.modules.module import Module


class Permute(Module):
    def __init__(self, permute_to: Tuple):
        """Permute Module.

        Args:
            permute_to (Tuple): Tuple with the target axis order.
        """
        super().__init__()
        self.permute_to = permute_to

    def compute_output_shape(self, input_shape):
        inputs = jax.core.ShapedArray((1, *input_shape[1:]), dtype=jnp.float32)
        shape = jax.eval_shape(self.forward, self.parameters(), inputs).shape
        return (None, *shape[1:])

    def forward(self, weights, inputs):
        """Permutes the inputs.

        Args:
            weights (Dict): Dictionary with attribute
                            names as keys and weights as values
            inputs (ndarray): len(permute_to)-D Tensor with
                              order axis (batch_size, *)

        Returns:
            ndarray: len(permute_to)-D Tensor
        """
        return F.permute(weights, inputs, self.permute_to)
