from typing import Dict

import jax
from jax.nn import initializers

from hera import backend
from hera.nn.modules import functional as F
from hera.nn.modules.module import Module


class LayerNormalization(Module):
    def __init__(
        self,
        normalized_shape: int,
        rng: int,
        scale: bool = True,
        center: bool = True,
        eps: float = 1e-05,
    ):
        """Layer Normalization Module

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
        super().__init__(rng, requires_rng=True)

        self.normalized_shape = normalized_shape

        # Whether to use trainable weights or just normalize without using them
        self.scale = scale
        self.center = center

        # Used to avoid dividing by zero
        self.eps = eps

        gamma_key, beta_key = backend.create_keys(self.rng, 2)
        if self.scale:
            self.add_weight(gamma_key, initializers.ones, (self.normalized_shape,), 'gamma')
        if self.center:
            self.add_weight(beta_key, initializers.zeros, (self.normalized_shape,), 'beta')

    def forward(self, weights: Dict, inputs: jax.numpy.ndarray) -> jax.numpy.ndarray:
        """Applies layer normalization over the feature dimension.

        Args:
            weights: (Dict): Dictionary with attribute names as keys and weights as values.
            inputs (jax.numpy.ndarray): Tensor with shape (*, normalized_shape)

        Returns:
            jax.numpy.ndarray: Tensor with shape (*, normalized_shape)
        """
        out = F.layer_normalization(
            inputs, weights["gamma"], weights["beta"], self.eps
        )
        return out
