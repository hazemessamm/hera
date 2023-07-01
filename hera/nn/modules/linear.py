from typing import Callable, Dict, Optional, Union

import jax
from jax.nn import initializers
from jax.random import PRNGKey

from hera import backend
from hera.nn.modules import functional as F
from hera.nn.modules.module import Module


class Linear(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rng: Optional[int] = None,
        activation: Callable = None,
        use_bias: bool = True,
    ):
        """Linear Layer which applies linear transformation on the inputs.

        >>> In case of initializing `global_rng`, the
        >>> hera.set_global_rng(5)
        >>> model = hera.nn.Linear(input_dim=10, output_dim=10)
        >>> model(model.parameters(), x)
        >>> # In case of not initializing `global_rng` (hera.set_global_rng)
        >>> # rng parameter should have a value.
        >>> model = nn.Linear(input_dim=10, output_dim=10, rng=10)
        >>> model(model.parameters(), x)


        Args:
            input_dim (int): Size of the input dimension.
            output_dim (int): Size of the output dimension.
            rng (int, optional): Seed or a random number that will 
                                 be used to create the weights.
                                 Default is None in case of creating a global rng,
                                 otherwise rng should be initialized.
            activation (Callable, optional): An activation function that will
                                             be called after the linear
                                             transformation. Defaults to None.
            use_bias (bool, optional): Whether to use the bias or not.
                                       Defaults to True.
        """
        super().__init__(rng, requires_rng=True)
        self.activation = activation
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.use_bias = use_bias

        # Create 2 keys for the weights and bias
        weight_key, bias_key = backend.create_keys(self.rng, 2)

        self.add_weight(weight_key, initializers.glorot_uniform(), (self.input_dim, self.output_dim), 'weight')
        if self.use_bias:
            self.add_weight(bias_key, initializers.zeros, (self.output_dim,), 'bias')


    def forward(
        self, weights: Dict, inputs: jax.numpy.ndarray
    ) -> jax.numpy.ndarray:
        """Applies linear transformation on the inputs.

        Args:
            weights: (Dict): Dictionary with attribute names as keys and weights as values.
            inputs (ndarray): A tensor with shape (*, input_dim)

        Returns:
            ndarray: A linearly transformed input with shape (*, output_dim).
        """

        if self.use_bias:
            bias = weights["bias"]
        else:
            bias = None

        out = F.linear(inputs, weights["weight"], bias=bias)

        if self.activation is not None:
            out = self.activation(out)

        return out
