from typing import Callable, List, Tuple, Union

from jax import lax
from jax.nn import initializers
from jax.numpy import DeviceArray

from hera.nn.modules import functional as F
from hera.nn.modules.module import Module
from hera.nn.modules.parameter import Parameter


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

        self._dimensions_spec = ("NHWC", "HWIO", "NHWC")

        self._validate_init()

        k1, k2 = self.create_keys(2)

        kernel_shape = (*self.kernel_size, in_channels, out_channels)
        self.weight = Parameter(k1, initializers.glorot_uniform(), kernel_shape)

        if self.use_bias:
            bias_shape = (self.out_channels,)
            self.bias = Parameter(k2, initializers.zeros, bias_shape)

        # self._dn = lax.conv_dimension_numbers(
        #     (1, 1, 1, in_channels), kernel_shape, self._dimensions_spec
        # )

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.reset_parameter()
        if self.use_bias:
            self.bias.reset_parameter()

    def _validate_init(self):
        if isinstance(self.kernel_size, tuple):
            if 0 >= len(self.kernel_size) > 2:
                raise ValueError(
                    "`kernel_size` should be a tuple with two "
                    "elements or an integer"
                )
        elif isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

        if isinstance(self.strides, tuple):
            if 0 >= len(self.strides) > 2:
                raise ValueError(
                    "`strides` should be a tuple with "
                    "two elements or an integer"
                )
        elif isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)

        if self.padding == "causal":
            raise ValueError(
                "`causal` padding is only allowed in "
                f"`Conv1D` module. Recieved padding={self.padding}"
            )

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        if len(input_shape) != 4:
            raise ValueError(
                f"`input_shape` should be a tuple with len(input_shape) == 4."
                f"Recieved {input_shape}"
            )

        return lax.conv_general_shape_tuple(
            lhs_shape=input_shape,
            rhs_shape=self.weight.shape,
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=self._dimensions_spec,
        )

    def forward(self, inputs: DeviceArray):
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

        if self.bias is None:
            bias = None
        else:
            bias = self.bias.data

        out = F.conv2d(
            inputs,
            self.weight.data,
            bias=bias,
            strides=self.strides,
            padding=self.padding,
        )

        if self.activation:
            out = self.activation(out)

        return out
