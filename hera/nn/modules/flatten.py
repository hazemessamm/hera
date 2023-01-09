from hera.nn.modules import functional as F
from hera.nn.modules.module import Module


class Flatten(Module):
    def __init__(self):
        """Flatten Module."""
        super().__init__()

    def forward(self, inputs):
        """Flattens the inputs.

        Args:
            weights (Dict): Dictionary with attribute
                            names as keys and weights as values
            inputs (ndarray): n+1-D Tensor with order axis (batch_size, *)

        Returns:
            ndarray: Flattened Tensor
        """
        return F.flatten(inputs)
