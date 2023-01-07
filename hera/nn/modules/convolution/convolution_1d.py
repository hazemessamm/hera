from typing import Callable, Dict, List, Tuple, Union

from jax import lax
from jax.nn import initializers
from jax.numpy import ndarray

from hera.nn.modules import functional as F
from hera.nn.modules.module import Module
from hera.nn.modules.parameter import Parameter


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

        self._dimensions_spec = ("NHC", "HIO", "NHC")

        self._validate_init()
        k1, k2 = self.create_keys(2)
        kernel_shape = (*self.kernel_size, in_channels, self.out_channels)
        self.weight = Parameter(k1, initializers.glorot_uniform(), kernel_shape)

        if self.use_bias:
            bias_shape = (self.out_channels,)
            self.bias = Parameter(k2, initializers.zeros, bias_shape)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.reset_parameter()
        if self.use_bias:
            self.bias.reset_parameter()

    def _validate_init(self):
        if isinstance(self.kernel_size, tuple):
            if 0 >= len(self.kernel_size) > 1:
                msg = """`kernel_size` should be a tuple with one element
                        or an integer"""
                raise ValueError(msg)
        elif isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,)

        if isinstance(self.strides, tuple):
            if 0 >= len(self.strides) > 1:
                raise ValueError(
                    "`strides` should be a tuple with one element or an integer"
                )
        elif isinstance(self.strides, int):
            self.strides = (int(self.strides),)

        if self.padding == "causal":
            raise ValueError("Causal padding is currently not implemented.")

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        if len(input_shape) != 3:
            raise ValueError("`input_shape` should be a tuple "
                             "with len(input_shape) == 4. "
                             f"Recieved {input_shape}")

        return lax.conv_general_shape_tuple(
            lhs_shape=input_shape,
            rhs_shape=self.weight.shape,
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=self._dimensions_spec,
        )

    def forward(self, weights: Dict, inputs: ndarray):
        """Applies convolution operation on inputs.

        Args:
            weights (Dict): Dictionary containing attribute names as keys
                            and weights as values.
            inputs (ndarray): A 3D tensor containing inputs with axis order:
                              (batch_size, timesteps, in_channels).

        Returns:
            ndarray: A 3D tensor with axis order:
                     (batch_size, timesteps, out_channels)
        """
        if self.use_bias:
            weights, bias = weights["weight"], weights["bias"]
        else:
            weights = weights["weight"]
            bias = None

        output = F.conv1d(
            inputs,
            weights,
            bias=bias,
            strides=self.strides,
            padding=self.padding,
        )

        return output
