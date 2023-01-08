from typing import Dict

import jax
from jax.numpy import ndarray

from hera.nn.modules.attention.multi_head_attention import MultiHeadAttention
from hera.nn.modules.dropout.dropout import Dropout
from hera.nn.modules.linear import Linear
from hera.nn.modules.module import Module
from hera.nn.modules.normalization.layer_normalization import \
    LayerNormalization
from hera.nn.modules.sequential import Sequential


class TransformerEncoderLayer(Module):
    """_summary_

    Args:
        Module (_type_): _description_
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        rng,
        intermediate_dim=512,
        attn_dropout=0.1,
        ff_dropout=0.1,
        ff_activation=jax.nn.gelu,
        jit=False,
    ):
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
            embed_dim,
            num_heads,
            rng=mha_key,
            dropout=attn_dropout,
            use_causal_mask=False,
        )
        self.layernorm_1 = LayerNormalization(embed_dim, rng=layernorm_1_key)

        self.ff = Sequential(
            [
                Linear(
                    embed_dim,
                    intermediate_dim,
                    rng=ff_key_1,
                    activation=ff_activation,
                ),
                Dropout(ff_dropout, rng=ff_dropout_key),
                Linear(intermediate_dim, embed_dim, rng=ff_key_2),
            ],
            jit=jit,
        )
        self.layernorm_2 = LayerNormalization(embed_dim, rng=layernorm_2_key)

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
            inputs (_type_): 3D tensor with axis order:
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
