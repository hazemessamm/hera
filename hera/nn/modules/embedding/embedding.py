from typing import Dict, Optional

import jax
from jax.nn import initializers

from hera import backend
from hera.nn.modules import functional as F
from hera.nn.modules.module import Module


class Embedding(Module):
    def __init__(
        self,
        embedding_size: int,
        embedding_dim: int,
        rng: Optional[int] = None,
        padding_idx: int = None,
    ):
        """Embedding Module.

        Args:
            embedding_size (int): Size of the embedding table.
            embedding_dim (int): Dimension size of each entry
                                 in the embedding table
            rng (int, optional): Seed for creating the embedding initial weights.
            padding_idx (int, optional): Zereos out the `padding_idx`
                                         if it is not `None`. Defaults to None.
        """
        super().__init__(rng)
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        key = backend.create_keys(self.rng, 1)
        self.add_weight(key, initializers.uniform(), (embedding_size, embedding_dim), 'weight')

    def forward(self, weights: Dict, inputs: jax.numpy.ndarray) -> jax.numpy.ndarray:
        """Returns the embedding of each index in the inputs.

        Args:
            weights: (Dict): Dictionary with attribute names as keys and weights as values.
            inputs (jax.numpy.ndarray): A n-D tensor of indices.

        Returns:
            jax.numpy.ndarray: A n+1-D tensor with shape (*, embed_dim).
        """
        out = F.embedding(inputs, weights["weight"])
        return out
