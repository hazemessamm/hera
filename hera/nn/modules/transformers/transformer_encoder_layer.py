from typing import Dict

import jax
from jax.numpy import ndarray

from hera.nn.modules.attention.multi_head_attention import MultiHeadAttention
from hera.nn.modules.dropout.dropout import Dropout
from hera.nn.modules.linear import Linear
from hera.nn.modules.module import Module
from hera.nn.modules.normalization.layer_normalization import LayerNormalization
from hera.nn.modules.sequential import Sequential
from typing import Callable


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        rng: int,
        intermediate_dim: int = 512,
        attn_dropout: int = 0.1,
        ff_dropout: int = 0.1,
        ff_activation: Callable = jax.nn.gelu,
        jit: bool = False,
    ):
        """Transformer Encoder Layer Module.

        Args:
            embedding_dim (int): Embedding dimension.
            num_heads (int): Number of heads,
                             must be divisible by the embedding_dim.
            rng (int): Seed for creating weights.
            intermediate_dim (int, optional): Intermediate dimension size for
                                              the feedforward module.
                                              Defaults to 512.
            attn_dropout (int, optional): Attention dropout rate.
                                          Defaults to 0.1.
            ff_dropout (int, optional): Feedforward dropout rate.
                                        Defaults to 0.1.
            ff_activation (Callable, optional): Feedforward activation function.
                                                Defaults to jax.nn.gelu.
            jit (bool, optional): Enables JIT compilation for
                                  the nested modules. Defaults to False.
        """
        super().__init__(rng, jit=jit)
        (
            mha_key,
            layernorm_1_key,
            ff_key_1,
            ff_key_2,
            layernorm_2_key,
            ff_dropout_key,
        ) = self.create_keys(6)

        self.mha = MultiHeadAttention(
            embedding_dim,
            num_heads,
            rng=mha_key,
            dropout=attn_dropout,
            use_causal_mask=False,
        )
        self.layernorm_1 = LayerNormalization(
            embedding_dim, rng=layernorm_1_key
        )

        self.ff = Sequential(
            [
                Linear(
                    embedding_dim,
                    intermediate_dim,
                    rng=ff_key_1,
                    activation=ff_activation,
                ),
                Dropout(ff_dropout, rng=ff_dropout_key),
                Linear(intermediate_dim, embedding_dim, rng=ff_key_2),
            ],
            jit=jit,
        )
        self.layernorm_2 = LayerNormalization(
            embedding_dim, rng=layernorm_2_key
        )

    def compute_output_shape(self, input_shape):
        inputs = jax.core.ShapedArray(
            (1, *input_shape[1:]), dtype=jax.numpy.float32
        )
        shape = jax.eval_shape(self.forward, self.parameters(), inputs).shape
        return (None, *shape[1:])

    def forward(self, weights: Dict, inputs: ndarray):
        """Passes the input through a transformer layer.

        Args:
            weights (Dict): Dictionary with attribute names as keys
                            and weights as values.
            inputs (ndarray): 3D tensor with axis order:
                             (batch_size, timesteps, embedding_dim).

        Returns:
            ndarray: 3D tensor with axis order:
                     (batch_size, timesteps, embedding_dim).
        """
        out = self.mha(weights["mha"], inputs, inputs, inputs)
        out_res = self.layernorm_1(weights["layernorm_1"], out + inputs)
        out = self.ff(weights["ff"], out_res)
        out = self.layernorm_2(weights["layernorm_2"], out + out_res)
        return out
