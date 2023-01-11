import jax
from jax import numpy as jnp
from jax.numpy import ndarray

from hera.nn.modules import functional as F
from hera.nn.modules.regularization.dropout import Dropout
from hera.nn.modules.linear import Linear
from hera.nn.modules.module import Module


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
            ValueError: If the embedding dimension is not divisble by
                        the number of heads.
        """
        super().__init__(rng=rng, jit=jit)

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim should be divisble by num_heads")

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.embed_dim_per_head = embed_dim // num_heads
        self.use_causal_mask = use_causal_mask
        self.dropout = dropout

        # Generate 4 keys for the 4 modules defined below.
        q_key, k_key, v_key, o_key, d_key = self.create_keys(5)
        self.q_proj = Linear(embed_dim, embed_dim, q_key, use_bias=use_bias)
        self.k_proj = Linear(embed_dim, embed_dim, k_key, use_bias=use_bias)
        self.v_proj = Linear(embed_dim, embed_dim, v_key, use_bias=use_bias)
        self.o_proj = Linear(embed_dim, embed_dim, o_key, use_bias=use_bias)
        self.attn_dropout = Dropout(dropout, d_key)

    def forward(self, query: ndarray, key: ndarray, value: ndarray):
        """Applies mutli head dot product attention.

        Args:
            query (ndarray): 3D tensor with shape
                             (batch_size, timesteps, embed_dim).
            key (ndarray): 3D tensor with shape
                           (batch_size, timesteps, embed_dim).
            value (ndarray): 3D tensor with shape
                             (batch_size, timesteps, embed_dim).

        Returns:
            ndarray: 3D tensor with shape (batch_size, timesteps, embed_dim).
        """
        # Pass `query`, `key` and `values` to the dense modules.
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        query, key, value = F.transpose_qkv(
            query, key, value, self.num_heads, self.embed_dim_per_head
        )
        scores = F.attention(query, key, self.use_causal_mask)
        scores = self.attn_dropout(scores)
        scores = F.score_value_matmul_and_transpose_scores(
            scores, value, self.embed_dim
        )
        out = self.o_proj(scores)
        return out

    def forward_with_external_weights(
        self, weights, query: ndarray, key: ndarray, value: ndarray
    ):
        """Applies mutli head dot product attention.

        Args:
            query (ndarray): 3D tensor with shape
                             (batch_size, timesteps, embed_dim).
            key (ndarray): 3D tensor with shape
                           (batch_size, timesteps, embed_dim).
            value (ndarray): 3D tensor with shape
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
