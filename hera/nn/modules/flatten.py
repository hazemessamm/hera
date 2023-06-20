from hera.nn.modules import functional as F
from hera.nn.modules.module import Module


class Flatten(Module):
    def __init__(self):
        """Flatten Module."""
        super().__init__(requires_rng=False)

    def forward(self, inputs):
        """Flattens the inputs.

        Args:
            inputs (ndarray): n+1-D Tensor with shape (batch_size, *)

        Returns:
            ndarray: Flattened Tensor
        """
        return F.flatten(inputs)
