from typing import Tuple

from hera.nn.modules import functional as F
from hera.nn.modules.module import Module


class Permute(Module):
    def __init__(self, target_shape: Tuple):
        """Permute Module.

        Args:
            target_shape (Tuple): Tuple with the target axis order.
        """
        super().__init__(requires_rng=False)
        self.target_shape = target_shape

    def forward(self, inputs: jax.numpy.ndarray) -> jax.numpy.ndarray:
        """Permutes the inputs.

        Args:
            inputs (jax.numpy.ndarray): len(target_shape)-D Tensor.

        Returns:
            jax.numpy.ndarray: len(target_shape)-D Tensor.
        """w
        return F.permute(inputs, self.target_shape)
