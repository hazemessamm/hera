import jax
from jax import numpy as jnp

from hera.nn.modules import functional as F
from hera.nn.modules.linear import Linear
from hera.nn.modules.dropout.dropout import Dropout
from hera.nn.modules.module import Module
from typing import Dict
from jax.numpy import ndarray


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
            jit (bool, optional): Whether to JIT compile the forward
                                  method or not. Defaults to False.

        Raises:
            ValueError: If the embedding dimension is not divisble by
                        the number of heads.
        """
        super().__init__(rng)

        # `use_causal_mask` is
        # inspired from
        # https://www.tensorflow.org/api_docs/python/tf/keras/modules/MultiHeadAttention

        if embed_dim % num_heads == 0:
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

    def compute_output_shape(self, input_shape):
        inputs = jax.core.ShapedArray((1, *input_shape[1:]), dtype=jnp.float32)
        shape = jax.eval_shape(self.forward, self.parameters(), inputs)
        return (None, *shape[1:]).shape

    def forward(
        self, weights: Dict, query: ndarray, key: ndarray, value: ndarray
    ):
        """Applies mutli head dot product attention.

        Args:
            weights (Dict): Dictionary where the keys are the nested modules
                            attribute names and their weights as values.
            query (ndarray): 3D tensor with shape
                             (batch_size, timesteps, embed_dim).
            key (ndarray): 3D tensor with shape
                           (batch_size, timesteps, embed_dim).
            value (ndarray): 3D tensor with shape
                             (batch_size, timesteps, embed_dim).

        Returns:
            ndarray: 3D tensor with shape (batch_size, timesteps, embed_dim).
        """
        batch_size = query.shape[0]

        # Pass `query`, `key` and `values` to the dense modules.
        query = self.q_proj(weights["q_proj"], query)
        key = self.k_proj(weights["k_proj"], key)
        value = self.v_proj(weights["v_proj"], value)

        # Reshape `query`, `key` and `values`
        # from shape [batch_size, seq_len, embed_dim]
        # to [batch_size, seq_len, num_heads, embed_dim // num_heads]
        # then change the `num_heads` axis from
        # [batch_size, seq_len, num_heads, embed_dim // num_heads]
        # to [batch_size, num_heads, seq_len, embed_dim // num_heads]
        # using the `transpose` function
        # and the reason for that is that
        # we want to process each head independetly
        # we want to have each head to have [seq_len, embed_dim // num_heads]
        query = jnp.reshape(
            query, (batch_size, -1, self.num_heads, self.embed_dim_per_head)
        ).transpose((0, 2, 1, 3))
        key = jnp.reshape(
            key, (batch_size, -1, self.num_heads, self.embed_dim_per_head)
        ).transpose((0, 2, 1, 3))
        value = jnp.reshape(
            value, (batch_size, -1, self.num_heads, self.embed_dim_per_head)
        ).transpose((0, 2, 1, 3))

        # Change the key axis from
        # [batch_size, num_heads, seq_len, embed_dim // num_heads]
        # to [batch_size, num_heads, embed_dim // num_heads, seq_len]
        # and the reason for that is because we need to be able
        # to perform matrix multiplication and in matrix multiplication
        # we need the first matrix columns to match the second matrix rows
        # in our situation we want query shape to be
        # [batch_size, num_heads, seq_len, embed_dim // num_heads]
        # and key shape to be
        # [batch_size, num_heads, embed_dim // num_heads, seq_len]
        # so we can have the last two axis matching each other.
        mul_key = key.transpose(0, 1, 3, 2)

        # Scale the scores output by the square root
        # of the embed_dim // num_heads (`embed_dim_per_head`)
        # to avoid having large variance
        scores = jnp.matmul(query, mul_key) / jnp.sqrt(self.embed_dim_per_head)

        # Apply causal mask if it's set to `True`
        if self.use_causal_mask:
            causal_mask = F.create_causal_mask(scores)
            scores = jnp.select(
                causal_mask, scores, jax.lax.broadcast(-jnp.inf, scores.shape)
            )

        # Transform the scores into probabilities.
        scores = jax.nn.softmax(scores, axis=-1)

        # Dropout the scores (regularizing the scores)
        # to avoid having high dependence on words.
        scores = self.attn_dropout(weights["attn_dropout"], scores)

        scores = jnp.matmul(scores, value)
        # Changing back the `out` shape from
        # [batch_size, num_heads, seq_len, embed_dim // num_heads]
        # to [batch_size, seq_len, num_heads, embed_dim // num_heads]
        # so we can collapse back the `num_heads` and `embed_dim // num_heads`
        # together back to `embed_dim`
        scores = jnp.transpose(scores, (0, 2, 1, 3))
        scores = jnp.reshape(scores, (batch_size, -1, self.embed_dim))
        # Pass the outputs to the final output dense module.
        out = self.o_proj(weights["o_proj"], scores)
        return out

    def __repr__(self):
        return f"""{self.__class__.__name__}(embed_dim={self.embed_dim},
                    num_heads={self.num_heads}, dropout={self.dropout},
                    use_causal_mask={self.use_causal_mask})"""
