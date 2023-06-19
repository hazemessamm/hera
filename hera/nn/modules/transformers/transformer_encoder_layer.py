from typing import Callable, Dict

import jax
from jax.numpy import ndarray

from hera.nn.modules.attention.multi_head_attention import MultiHeadAttention
from hera.nn.modules.linear import Linear
from hera.nn.modules.module import Module
from hera.nn.modules.normalization.layer_normalization import \
    LayerNormalization
from hera.nn.modules.regularization.dropout import Dropout
from hera.nn.modules.sequential import Sequential
from hera import backend


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
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.ff_activation = ff_activation
        (
            mha_key,
            layernorm_1_key,
            ff_key_1,
            ff_key_2,
            layernorm_2_key,
            ff_dropout_key,
        ) = backend.create_keys(self.rng, 6)

        self.mha = MultiHeadAttention(
            self.embedding_dim,
            self.num_heads,
            rng=mha_key,
            dropout=self.attn_dropout,
            use_causal_mask=False,
        )
        self.layernorm_1 = LayerNormalization(
            self.embedding_dim, rng=layernorm_1_key
        )

        self.ff = Sequential(
            [
                Linear(
                    self.embedding_dim,
                    self.intermediate_dim,
                    rng=ff_key_1,
                    activation=self.ff_activation,
                ),
                Dropout(self.ff_dropout, rng=ff_dropout_key),
                Linear(self.intermediate_dim, self.embedding_dim, rng=ff_key_2),
            ],
            jit=False,
        )
        self.layernorm_2 = LayerNormalization(
            self.embedding_dim, rng=layernorm_2_key
        )



    # def build(self):
    #     (
    #         mha_key,
    #         layernorm_1_key,
    #         ff_key_1,
    #         ff_key_2,
    #         layernorm_2_key,
    #         ff_dropout_key,
    #     ) = backend.create_keys(self.rng, 6)

    #     self.mha = MultiHeadAttention(
    #         self.embedding_dim,
    #         self.num_heads,
    #         rng=mha_key,
    #         dropout=self.attn_dropout,
    #         use_causal_mask=False,
    #     )
    #     self.layernorm_1 = LayerNormalization(
    #         self.embedding_dim, rng=layernorm_1_key
    #     )

    #     self.ff = Sequential(
    #         [
    #             Linear(
    #                 self.embedding_dim,
    #                 self.intermediate_dim,
    #                 rng=ff_key_1,
    #                 activation=self.ff_activation,
    #             ),
    #             Dropout(self.ff_dropout, rng=ff_dropout_key),
    #             Linear(self.intermediate_dim, self.embedding_dim, rng=ff_key_2),
    #         ],
    #         jit=False,
    #     )
    #     self.layernorm_2 = LayerNormalization(
    #         self.embedding_dim, rng=layernorm_2_key
    #     )
    #     self.built = True


    def forward(self, weights: Dict, inputs: ndarray):
        """Passes the input through a transformer layer.

        Args:
            weights: (Dict): Dictionary with attribute names as keys and weights as values.
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
