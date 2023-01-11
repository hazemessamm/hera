from jax.numpy import ndarray
from nn.modules import functional as F
from nn.modules.embedding.embedding import Embedding


class PositionalEmbedding(Embedding):
    def __init__(self, max_length: int, embedding_dim: int, rng: int):
        """Learnable Positional Embedding Module.

        Args:
            max_seq_len (int): Maximum timesteps per example.
            embedding_dim (int): Embedding dimension per position.
            rng (int): Seed for creating initial weights.
        """
        super().__init__(max_length, embedding_dim, rng)

    def forward(self, inputs: ndarray):
        """Creates positions tensors and returns their embeddings.

        Args:
            inputs (ndarray): Tensor of indices.

        Returns:
            ndarray: A n+1-D tensor with order axis: (*, embed_dim).
        """
        out = F.positional_embedding(inputs, self.weight.data)
        return out

    def forward_with_external_weights(self, weights, inputs: ndarray):
        """Creates positions tensors and returns their embeddings.

        Args:
            inputs (ndarray): Tensor of indices.

        Returns:
            ndarray: A n+1-D tensor with order axis: (*, embed_dim).
        """
        out = F.positional_embedding(inputs, weights["weight"])
        return out
