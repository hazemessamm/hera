import jax
from jax import lax

from hera.nn.modules import functional as F
from hera.nn.modules.module import Module


class AvgPooling2D(Module):
    def __init__(self, pool_size, strides=None, padding="valid"):
        """Average Pooling Module."""
        super().__init__(requires_rng=False)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def forward(self, inputs: jax.numpy.ndarray) -> jax.numpy.ndarray:
        """Applies average pooling for spatial data.

        Args:
            inputs (jax.numpy.ndarray): A 4D tensor with shape (batch_size, height, width, channels)

        Returns:
            jax.numpy.ndarray: A 4D tensor with shape (batch_size, height, width, channels)
        """
        if isinstance(self.padding, tuple):
            padding = self.padding
        elif isinstance(self.padding, str):
            padding = tuple(lax.padtype_to_pads(inputs.shape, self.pool_size, self.strides, self.padding))
            self.padding = padding
        else:
            padding = tuple(self.padding)
            self.padding = padding
        
        return F.average_pooling_2d(inputs, pool_size=self.pool_size, strides=self.strides, padding=padding)
