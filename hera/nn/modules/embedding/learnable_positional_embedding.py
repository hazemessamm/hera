from typing import Dict

from jax.numpy import ndarray
from hera.nn.modules import functional as F
from hera.nn.modules.embedding.embedding import Embedding


class PositionalEmbedding(Embedding):
    def __init__(self, max_length: int, embedding_dim: int, rng: int):
        """Learnable Positional Embedding Module.

        Args:
            max_seq_len (int): Maximum timesteps per example.
            embedding_dim (int): Embedding dimension per position.
            rng (int): Seed for creating initial weights.
        """
        super().__init__(max_length, embedding_dim, rng)

    def forward(self, weights: Dict, inputs: ndarray):
        """Creates positions tensors and returns their embeddings.

        Args:
            weights: (Dict): Dictionary with attribute names as keys and weights as values.
            inputs (ndarray): Tensor of indices.

        Returns:
            ndarray: A n+1-D tensor with order axis: (*, embed_dim).
        """
        out = F.positional_embedding(inputs, weights["weight"])
        return out
