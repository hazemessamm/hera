from typing import Callable, Dict, Union

import jax
from jax import numpy as jnp
from jax.nn import initializers
from jax.numpy import ndarray
from jax.random import PRNGKey
from nn.modules import functional as F
from nn.modules.module import Module
from nn.modules.parameter import Parameter


class Linear(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rng: Union[int, PRNGKey],
        activation: Callable = None,
        use_bias: bool = True,
    ):
        """Linear Layer which applies linear transformation on the inputs.

        Args:
            input_dim (int): Size of the input dimension.
            output_dim (int): Size of the output dimension.
            rng (int): Seed or a random number that will
                       be used to create the weights.
            activation (Callable, optional): An activation function that will
                                             be called after the linear
                                             transformation. Defaults to None.
            use_bias (bool, optional): Whether to use the bias or not.
                                       Defaults to True.
        """
        super().__init__(rng)
        self.activation = activation
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.use_bias = use_bias

        # Create 2 subkeys for the weights and bias
        weight_key, bias_key = self.create_keys(2)
        self.weight = Parameter(
            weight_key, initializers.glorot_uniform(), (input_dim, output_dim)
        )
        if use_bias:
            self.bias = Parameter(bias_key, initializers.zeros, (output_dim,))
        self.reset_parameters()

    def compute_output_shape(self, input_shape):
        inputs = jax.core.ShapedArray((1, *input_shape[1:]), dtype=jnp.float32)
        shape = jax.eval_shape(self.forward, self.parameters(), inputs).shape
        return (None, *shape[1:])

    def reset_parameters(self):
        """Resets (re-intiialize or initialize) Linear module weights.
        """
        self.weight.reset_parameter()
        if self.use_bias:
            self.bias.reset_parameter()

    def forward(self, weights: Dict, inputs: ndarray) -> ndarray:
        """Applies linear transformation on the inputs.

        Args:
            weights (Dict): A dictionary containing the weights
                            with `weights` and `bias` as keys.
            inputs (ndarray): A tensor with shape (*, input_dim)

        Returns:
            ndarray: A linearly transformed input with shape (*, output_dim).
        """
        weight = weights["weight"]
        if self.use_bias:
            bias = weights["bias"]
        else:
            bias = None

        out = F.linear(inputs, weight, bias=bias)

        if self.activation is not None:
            out = self.activation(out)

        return out

    def __repr__(self):
        return f"""{self.__class__.__name__}(input_dim={self.input_dim},
                    output_dim={self.output_dim}, activation={self.activation})
                """
