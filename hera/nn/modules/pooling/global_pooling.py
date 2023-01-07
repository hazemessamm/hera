import jax
from jax import numpy as jnp

from nn.modules import functional as F
from nn.modules.module import Module
from typing import Dict
from jax.numpy import ndarray


class GlobalMaxPooling1D(Module):
    def __init__(self, jit=False):
        """Global Max Pooling Module.

        Args:
            jit (bool, optional): JIT compiles `forward()` if enabled.
                                  Defaults to False.
        """
        super().__init__(jit=jit)

    def compute_output_shape(self, input_shape):
        inputs = jax.core.ShapedArray((1, *input_shape[1:]), dtype=jnp.float32)
        shape = jax.eval_shape(self.forward, self.parameters(), inputs).shape
        return (None, *shape[1:])

    def forward(self, weights: Dict, inputs: ndarray):
        """Applies global max pooling over the timesteps axis.

        Args:
            weights (Dict): Dictionary of attributes as keys
                            and weights as values.
            inputs (ndarray): A 3D tensor with
                              axis order: (batch_size, timesteps, features)

        Returns:
            ndarray: A 2D tensor with axis order: (batch_size, features)
        """
        return F.global_max_pooling_1d(inputs, weights)


class GlobalAvgPooling1D(Module):
    def __init__(self, jit=False):
        """Global Average Pooling Module.

        Args:
            jit (bool, optional): JIT compils `forward()` if eenabled.
                                  Defaults to False.
        """
        super().__init__(jit=jit)

    def compute_output_shape(self, input_shape):
        inputs = jax.core.ShapedArray((1, *input_shape[1:]), dtype=jnp.float32)
        shape = jax.eval_shape(self.forward, self.parameters(), inputs).shape
        return (None, *shape[1:])

    def forward(self, weights, inputs):
        """Applies global average pooling over the timesteps axis.

        Args:
            weights (Dict): Dictionary of attributes as keys
                            and weights as values.
            inputs (ndarray): A 3D tensor with
                              axis order: (batch_size, timesteps, features)

        Returns:
            ndarray: A 2D tensor with axis order: (batch_size, features)
        """
        return F.global_avg_pooling_1d(inputs, weights)
