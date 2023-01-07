import jax
from jax.nn import initializers

from nn.modules import functional as F
from nn.modules.module import Module
from nn.modules.parameter import Parameter
import jax.numpy as jnp
from typing import Dict
from jax.numpy import ndarray


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

    def compute_output_shape(self, input_shape):
        inputs = jax.core.ShapedArray((1, *input_shape[1:]), dtype=jnp.float32)
        shape = jax.eval_shape(self.forward, self.parameters(), inputs).shape
        return (None, *shape[1:])

    def reset_parameters(self):
        self.weight.reset_parameter()

    def forward(self, weights: Dict, inputs: ndarray):
        """Returns the embedding of each index in the inputs.

        Args:
            weights (Dict): dictionary with the attribute names as keys
                            and weights as values
            inputs (ndarray): A n-D tensor of indices

        Returns:
            ndarray: A n+1-D tensor with order axis: (*, embed_dim).
        """
        out = F.embedding(inputs, weights["weight"])
        return out

    def __repr__(self):
        return f"""{self.__class__.__name__}(vocab_size={self.vocab_size},
                embed_dim={self.embed_dim})"""
