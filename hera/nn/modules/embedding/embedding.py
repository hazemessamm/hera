from typing import Dict

from jax.nn import initializers
from jax.numpy import ndarray

from hera.nn.modules import functional as F
from hera.nn.modules.module import Module
from hera.nn.modules.parameter import Parameter


class Embedding(Module):
    def __init__(
        self,
        embedding_size: int,
        embedding_dim: int,
        rng: int,
        padding_idx: int = None,
    ):
        """Embedding Module.

        Args:
            embedding_size (int): Size of the embedding table.
            embedding_dim (int): Dimension size of each entry
                                 in the embedding table
            rng (int): Seed for creating the embedding initial weights.
            padding_idx (int, optional): Zereos out the `padding_idx`
                                         if it is not `None`. Defaults to None.
        """
        super().__init__(rng)
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        key = self.create_keys(1)
        self.weight = Parameter(
            key,
            initializers.uniform(),
            shape=(self.embedding_size, self.embedding_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.reset_parameter()

    def forward(self, inputs: ndarray):
        """Returns the embedding of each index in the inputs.

        Args:
            inputs (ndarray): A n-D tensor of indices.

        Returns:
            ndarray: A n+1-D tensor with order axis: (*, embed_dim).
        """
        out = F.embedding(inputs, self.weight.data)
        return out

    def forward_manual(self, weights: Dict, inputs: ndarray):
        """Returns the embedding of each index in the inputs.

        Args:
            weights: (Dict): Dictionary with attribute names as keys and weights as values.
            inputs (ndarray): A n-D tensor of indices.

        Returns:
            ndarray: A n+1-D tensor with order axis: (*, embed_dim).
        """
        out = F.embedding(inputs, weights["weight"])
        return out
