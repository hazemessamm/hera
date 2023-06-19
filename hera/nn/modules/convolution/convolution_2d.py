from typing import Callable, Dict, List, Tuple, Union, Optional

from jax import lax
from jax.nn import initializers
from jax.numpy import DeviceArray

from hera.nn.modules import functional as F
from hera.nn.modules.module import Module
from hera import backend
from hera.nn.modules.convolution import conv_validation


class Conv2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        rng: Optional[int] = None,
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
                       Default is None in case of creating a global rng,
                       otherwise rng should be initialized with an integer.
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

        super(Conv2D, self).__init__(rng=rng, requires_rng=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        conv_validation.validate_conv2d_init(self)

        self._dimensions_spec = ("NHWC", "HWIO", "NHWC")

        k1, k2 = backend.create_keys(self.rng, 2)
        self.add_weight(k1, initializers.glorot_uniform(), (*self.kernel_size, self.in_channels, self.out_channels), 'weight')
        if self.use_bias:
            self.add_weight(k2, initializers.zeros, (self.out_channels,), 'bias')

    # def build(self):
    #     k1, k2 = backend.create_keys(self.rng, 2)
    #     self.add_weight(k1, initializers.glorot_uniform(), (*self.kernel_size, self.in_channels, self.out_channels), 'weight')
    #     if self.use_bias:
    #         self.add_weight(k2, initializers.zeros, (self.out_channels,), 'bias')
    #     self.built = True
    
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

    def forward(self, weights: Dict, inputs: DeviceArray):
        """Applies convolution operation on inputs.

        Args:
            weights: (Dict): Dictionary with attribute names as keys and weights as values.
            inputs (ndarray): A 4D tensor containing inputs with axis order:
                              (batch_size, height, width, in_channels).

        Returns:
            ndarray: A 3D tensor with axis order:
                     (batch_size, height, width, out_channels)
        """
        if self.use_bias:
            bias = weights["bias"]
        else:
            bias = None

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
