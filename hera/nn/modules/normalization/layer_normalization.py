import jax
from jax.nn import initializers
from nn.modules import functional as F
from nn.modules.module import Module
from nn.modules.parameter import Parameter
from jax.numpy import ndarray
from typing import Dict
from jax import numpy as jnp


class LayerNormalization(Module):
    def __init__(
        self,
        normalized_shape: int,
        rng: int,
        scale: bool = True,
        center: bool = True,
        eps: float = 1e-05,
    ):
        """Layer normalization module

        Args:
            normalized_shape (int): Number of features to be normalized.
            rng (int): seed that will be used for initializing gamma and beta.
            scale (bool, optional): If set to `True` the inputs will be scaled.
                                    Defaults to True.
            center (bool, optional): If set to `True` the inputs will be
                                     centered. Defaults to True.
            eps (float, optional): Epsilon that is used to avoid division
                                   by zero. Defaults to 1e-05.
        """
        super().__init__(rng)

        self.normalized_shape = normalized_shape

        # Whether to use trainable weights or just normalize without using them
        self.scale = scale
        self.center = center

        # Used to avoid dividing by zero
        self.eps = eps

        gamma_key, beta_key = self.create_keys(2)
        if self.scale:
            self.gamma = Parameter(
                gamma_key, initializers.ones, shape=(self.normalized_shape,)
            )
        if self.center:
            self.beta = Parameter(
                beta_key, initializers.zeros, shape=(self.normalized_shape,)
            )

        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.reset_parameter()
        self.beta.reset_parameter()

    def compute_output_shape(self, input_shape):
        inputs = jax.core.ShapedArray((1, *input_shape[1:]), dtype=jnp.float32)
        shape = jax.eval_shape(self.forward, self.parameters(), inputs)
        return (None, *shape[1:]).shape

    def forward(self, weights: Dict, inputs: ndarray):
        """Applies layer normalization over the feature dimension.

        Args:
            weights (Dict): A dictionary with `gamma` and `beta` as keys
                            and their tensors as values.
            inputs (ndarray): Tensor with shape (*, normalized_shape)

        Returns:
            ndarray: Tensor with shape (*, normalized_shape)
        """
        if self.scale:
            gamma = weights["gamma"]
        if self.center:
            beta = weights["beta"]
        out = F.layer_normalization(inputs, gamma, beta, self.eps)
        return out