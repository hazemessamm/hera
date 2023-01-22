from typing import Callable, Dict, List, Tuple, Union

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

        if not self._reconstructed_from_unflatten:
            self._validate_init()

        k1, k2 = self.create_keys(2)

        self.weight = Parameter(rng=k1, initializer=initializers.glorot_uniform(), shape=(*self.kernel_size, in_channels, out_channels))

        if self.use_bias:
            self.bias = Parameter(rng=k2, initializer=initializers.zeros, shape=(self.out_channels,))

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

        if isinstance(self.strides, int):
            if self.strides <= 0:
                raise ValueError(f"`strides` should be a tuple with 2 values bigger than zero. Recieved {self.strides}")
            self.strides = (1, self.strides, self.strides, 1)
        elif isinstance(self.strides, tuple):
            if 1 <= len(self.strides) < 2:
                self.strides += self.strides
            elif len(self.strides) > 2 or len(self.strides) < 1:
                raise ValueError(f'`strides` should be a tuple with length of 2. Recieved {self.strides}')
        else:
            raise ValueError(f'Expected `strides` to be a tuple with length of 2 or an integer. Recieved {self.strides}')

        if any(s <= 0 for s in self.strides):
            raise ValueError(f'`strides` should be a tuple of values where each value should be bigger than or equal to 1. Recieved {self.strides}')
        
        if isinstance(self.padding, str):
            if self.padding.lower() not in {'valid', 'same'}:
                raise ValueError('`padding` should be a string with values'
                                 f'`valid` or `same` or a tuple with length of 2. '
                                 f'Recieved {self.padding}')
            
            self.padding = self.padding.upper()
        elif isinstance(self.padding, (list, tuple)):
            if not any(isinstance(p, tuple) for p in self.padding):
                raise ValueError(f'`padding` should be a list of 4 tuples where each tuple should contain two elements. Recieved {self.padding}')
        else:
            raise ValueError(f'Expected `padding` to be a `str`, `list` or `tuple` of 4 `tuples` where each `t`uple should contain two elements. Recieved {self.padding}')



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
            weights=self.weight.data,
            bias=bias,
            strides=self.strides,
            padding=self.padding,
        )

        if self.activation:
            out = self.activation(out)

        return out

    def forward_manual(self, weights: Dict, inputs: DeviceArray):
        """Applies convolution operation on inputs.

        Args:
            weights: (Dict): Dictionary with attribute names as keys and weights as values.
            inputs (ndarray): A 4D tensor containing inputs with axis order:
                              (batch_size, height, width, in_channels).

        Returns:
            ndarray: A 3D tensor with axis order:
                     (batch_size, height, width, out_channels)
        """

        if self.bias is None:
            bias = None
        else:
            bias = weights["bias"]

        out = F.conv2d(
            inputs,
            weights["weight"],
            bias=bias,
            strides=self.strides,
            padding=self.padding,
        )

        if self.activation:
            out = self.activation(out)

        return out
