from typing import Dict

import jax

from hera import backend
from hera.nn.modules import functional as F
from hera.nn.modules.linear import Linear
from hera.nn.modules.module import Module
from hera.nn.modules.regularization.dropout import Dropout


class MultiHeadAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rng: int,
        dropout: float = 0.0,
        use_causal_mask: bool = False,
        use_bias: bool = True,
        jit: bool = False,
    ):
        """Multihead Attention Module

        Args:
            embed_dim (int): Size of the embedding dimension.
            num_heads (int): Number of heads.
            rng (int): Seed for initializing the linear modules.
            dropout (float, optional): dropout probability. Defaults to 0.0.
            use_causal_mask (bool, optional): Whether to mask the future
                                              tokens or not. Defaults to False.
            use_bias (bool, optional): Whether to use the bias in the linear
                                       modules or not. Defaults to True.
            jit (bool, optional): JIT compiles `forward()` if enabled.
                                  Defaults to False.

        Raises:
            ValueError: If the embedding dimension is not divisible by
                        the number of heads.
        """
        super().__init__(rng=rng, jit=jit)

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim should be divisible by num_heads")

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.embed_dim_per_head = embed_dim // num_heads
        self.use_causal_mask = use_causal_mask
        self.dropout = dropout
        self.use_bias = use_bias

        self.q_proj = Linear(embed_dim, embed_dim, use_bias=use_bias)
        self.k_proj = Linear(embed_dim, embed_dim, use_bias=use_bias)
        self.v_proj = Linear(embed_dim, embed_dim, use_bias=use_bias)
        self.o_proj = Linear(embed_dim, embed_dim, use_bias=use_bias)
        self.attn_dropout = Dropout(dropout)


    def forward(
        self, weights: Dict, query: jax.numpy.ndarray, key: jax.numpy.ndarray, value: jax.numpy.ndarray
    ) -> jax.numpy.ndarray:
        """Applies mutli head dot product attention.

        Args:
            weights: (Dict): Dictionary with attribute names as keys and weights as values.
            query (jax.numpy.ndarray): 3D tensor with shape
                                       (batch_size, timesteps, embed_dim).
            key (jax.numpy.ndarray): 3D tensor with shape
                                     (batch_size, timesteps, embed_dim).
            value (jax.numpy.ndarray): 3D tensor with shape
                                       (batch_size, timesteps, embed_dim).

        Returns:
            ndarray: 3D tensor with shape (batch_size, timesteps, embed_dim).
        """
        # Pass `query`, `key` and `values` to the dense modules.
        query = self.q_proj(weights["q_proj"], query)
        key = self.k_proj(weights["k_proj"], key)
        value = self.v_proj(weights["v_proj"], value)
        query, key, value = F.transpose_qkv(
            query, key, value, self.num_heads, self.embed_dim_per_head
        )
        scores = F.attention(query, key, self.use_causal_mask)
        scores = self.attn_dropout(weights["attn_dropout"], scores)
        scores = F.score_value_matmul_and_transpose_scores(
            scores, value, self.embed_dim
        )
        out = self.o_proj(weights["o_proj"], scores)
        return out

    def __repr__(self):
        return f"""{self.__class__.__name__}(embed_dim={self.embed_dim},
                    num_heads={self.num_heads}, dropout={self.dropout},
                    use_causal_mask={self.use_causal_mask})"""
