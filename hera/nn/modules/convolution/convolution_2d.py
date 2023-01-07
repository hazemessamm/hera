from typing import Callable, Tuple, Union, List

from jax import lax
from jax.numpy import DeviceArray
from nn.modules.module import Module
from nn.modules.parameter import Parameter
from jax.nn import initializers
from nn.modules import functional as F
from typing import Dict


class Conv2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        rng: int,
        strides: Union[int, tuple] = (1, 1),
        padding: str = "valid",
        activation: Union[str, Callable] = None,
        use_bias: bool = True,
    ):
        """Conv2D Module

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Union[int, tuple]): Number of filters.
            rng (int): Seed for creating the weights and bias.
            strides (Union[int, tuple], optional): Number of strides.
                                                   Accepts integer or a tuple.
                                                   Defaults to (1, 1).
            padding (str, optional): padding type. Currently supports `valid`,
                                     `same` and a tuple with custom padding
                                     values. Defaults to "valid".
            activation (Callable, optional): Activation function to get called
                                             after the transformation.
                                             Defaults to None.
            use_bias (bool, optional): Whether to enable or disable the bias.
                                       Defaults to True.
        """

        super(Conv2D, self).__init__(rng=rng)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

        self._dimensions_spec = ("NHC", "HIO", "NHC")

        self._validate_init()

        k1, k2 = self.create_keys(2)
        kernel_shape = (*self.kernel_size, in_channels, self.filters)
        self.weight = Parameter(k1, initializers.glorot_uniform, kernel_shape)

        if self.use_bias:
            bias_shape = (self.filters,)
            self.bias = Parameter(k2, initializers.zeros, bias_shape)

        self._dn = lax.conv_dimension_numbers(
            (1, 1, in_channels), kernel_shape, self.dimensions_spec
        )

    def _validate_init(self):
        if isinstance(self.kernel_size, tuple):
            if 0 >= len(self.kernel_size) > 2:
                msg = """`kernel_size` should be a tuple with
                          two elements or an integer"""
                raise ValueError(msg)
        elif isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

        if isinstance(self.strides, tuple):
            if 0 >= len(self.strides) > 2:
                msg = """`strides` should be a tuple with two elements
                          or an integer"""
                raise ValueError(msg)
        elif isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)

        if self.padding == "causal":
            raise ValueError(
                f"""`causal` padding is only allowed in
                    `Conv1D` module. Recieved padding={self.padding}"""
            )

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        if len(input_shape) != 4:
            raise ValueError(
                f"""`input_shape` should be a tuple
                    with len(input_shape) == 4. Recieved {input_shape}"""
            )
        return lax.conv_general_shape_tuple(
            lhs_shape=input_shape,
            rhs_shape=self.weight.shape,
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=self.dimension_numbers,
        )

    def forward(self, weights: Dict, inputs: DeviceArray):
        """Applies convolution operation on inputs.

        Args:
            weights (Dict): Dictionary containing attribute names as keys
                            and weights as values.
            inputs (ndarray): A 4D tensor containing inputs with axis order:
                              (batch_size, height, width, in_channels).

        Returns:
            ndarray: A 3D tensor with axis order:
                     (batch_size, height, width, out_channels)
        """
        if self.use_bias:
            weights, bias = weights["weight"], weights["bias"]
        else:
            weights = weights["weight"]
            bias = None

        out = F.conv2d(
            inputs,
            weights,
            bias=bias,
            strides=self.strides,
            padding=self.padding,
        )

        if self.activation:
            out = self.activation(out)

        return out
