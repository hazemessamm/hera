import jax

from hera.nn.modules import functional as F
from hera.nn.modules.module import Module


class GlobalMaxPooling2D(Module):
    def __init__(self):
        """Global Max Pooling Module."""
        super().__init__(requires_rng=False)

    def forward(self, inputs: jax.numpy.ndarray) -> jax.numpy.ndarray:
        """Applies global max pooling over the timesteps axis.

        Args:
            inputs (jax.numpy.ndarray): A 3D tensor with shape (batch_size, timesteps, features)

        Returns:
            jax.numpy.ndarray: A 2D tensor with shape (batch_size, features)
        """
        return F.global_max_pooling_2d(inputs)


class GlobalAvgPooling2D(Module):
    def __init__(self):
        """Global Average Pooling Module."""
        super().__init__(requires_rng=False)

    def forward(self, inputs: jax.numpy.ndarray) -> jax.numpy.ndarray:
        """Applies global average pooling over the timesteps axis.

        Args:
            weights (Dict): Dictionary of attributes as keys
                            and weights as values.
            inputs (jax.numpy.ndarray): A 3D tensor with shape (batch_size, timesteps, features)

        Returns:
            jax.numpy.ndarray: A 2D tensor with shape (batch_size, features)
        """
        return F.global_avg_pooling_2d(inputs)
