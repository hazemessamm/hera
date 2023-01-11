from typing import Callable, Union

from jax.nn import initializers
from jax.numpy import ndarray
from jax.random import PRNGKey

from hera.nn.modules import functional as F
from hera.nn.modules.module import Module
from hera.nn.modules.parameter import Parameter


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

    def reset_parameters(self):
        """Resets (re-intiialize or initialize) Linear module weights."""
        self.weight.reset_parameter()
        if self.use_bias:
            self.bias.reset_parameter()

    def forward(self, inputs: ndarray) -> ndarray:
        """Applies linear transformation on the inputs.

        Args:
            inputs (ndarray): A tensor with axis order: (*, input_dim)

        Returns:
            ndarray: A linearly transformed input
                     with axis order: (*, output_dim).
        """

        if self.use_bias:
            bias = self.bias.data
        else:
            bias = None

        out = F.linear(inputs, self.weight.data, bias=bias)

        if self.activation is not None:
            out = self.activation(out)

        return out

    def forward_with_external_weights(
        self, weights, inputs: ndarray
    ) -> ndarray:
        """Applies linear transformation on the inputs.

        Args:
            inputs (ndarray): A tensor with axis order: (*, input_dim)

        Returns:
            ndarray: A linearly transformed input
                     with axis order: (*, output_dim).
        """

        if self.use_bias:
            bias = weights["bias"]
        else:
            bias = None

        out = F.linear(inputs, weights["weight"], bias=bias)

        if self.activation is not None:
            out = self.activation(out)

        return out
