from typing import Tuple

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
