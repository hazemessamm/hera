from typing import Callable, Dict, List, Tuple, Union

import jax
from jax import lax
from jax.nn import initializers

from hera import backend
from hera.nn.modules import functional as F
from hera.nn.modules.convolution import conv_validation
from hera.nn.modules.module import Module


class Conv1D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        rng: int,
        strides: Union[int, tuple] = (1,),
        padding: str = "valid",
        activation: Callable = None,
        use_bias: bool = True,
    ):
        """Conv1D Module

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Union[int, tuple]): Number of filters.
            rng (int): Seed for creating the weights and bias.
            strides (Union[int, tuple], optional): Number of strides.
                                                   Accepts integer or a tuple.
                                                   Defaults to (1,).
            padding (str, optional): padding type. Currently supports `valid`,
                                     `same` and a tuple with custom padding
                                     values. Defaults to "valid".
            activation (Callable, optional): Activation function to get called
                                             after the transformation.
                                             Defaults to None.
            use_bias (bool, optional): Whether to enable or disable the bias.
                                       Defaults to True.
        """
        super(Conv1D, self).__init__(rng=rng)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

        conv_validation.validate_conv1d_init(self)

        self._dimensions_spec = ("NHC", "HIO", "NHC")
        
        k1, k2 = backend.create_keys(self.rng, 2)
        kernel_shape = (*self.kernel_size, self.in_channels, self.out_channels)
        self.add_weight(k1, initializers.xavier_uniform(), kernel_shape, 'weight')

        if self.use_bias:
            bias_shape = (self.out_channels,)
            self.add_weight(k2, initializers.zeros, bias_shape, 'bias')
        else:
            self.bias = None

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        if len(input_shape) != 3:
            raise ValueError(
                "`input_shape` should be a tuple "
                "with len(input_shape) == 4. "
                f"Received {input_shape}"
            )

        return lax.conv_general_shape_tuple(
            lhs_shape=input_shape,
            rhs_shape=self.weight.shape,
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=self._dimensions_spec,
        )

    def forward(self, weights: Dict, inputs: jax.numpy.ndarray) -> jax.numpy.ndarray:
        """Applies convolution operation on inputs.

        Args:
            weights: (Dict): Dictionary with attribute names as keys and weights as values.
            inputs (jax.numpy.ndarray): A 3D tensor containing inputs with shape (batch_size, timesteps, in_channels).

        Returns:
            jax.numpy.ndarray: A 3D tensor with shape (batch_size, timesteps, out_channels)
        """

        if self.bias is None:
            bias = None
        else:
            bias = weights["bias"]

        output = F.conv1d(
            inputs,
            weights["weight"],
            bias=bias,
            strides=self.strides,
            padding=self.padding,
        )

        return output
