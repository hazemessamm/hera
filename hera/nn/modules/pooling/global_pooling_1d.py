from jax.numpy import ndarray

from hera.nn.modules import functional as F
from hera.nn.modules.module import Module


class GlobalMaxPooling1D(Module):
    def __init__(self):
        """Global Max Pooling Module."""
        super().__init__(requires_rng=False)

    def forward(self, inputs: ndarray):
        """Applies global max pooling over the timesteps axis.

        Args:
            inputs (ndarray): A 3D tensor with
                              axis order: (batch_size, timesteps, features)

        Returns:
            ndarray: A 2D tensor with axis order: (batch_size, features)
        """
        return F.global_max_pooling_1d(inputs)


class GlobalAvgPooling1D(Module):
    def __init__(self):
        """Global Average Pooling Module."""
        super().__init__(requires_rng=False)

    def forward(self, inputs):
        """Applies global average pooling over the timesteps axis.

        Args:
            weights (Dict): Dictionary of attributes as keys
                            and weights as values.
            inputs (ndarray): A 3D tensor with
                              axis order: (batch_size, timesteps, features)

        Returns:
            ndarray: A 2D tensor with axis order: (batch_size, features)
        """
        return F.global_avg_pooling_1d(inputs)
