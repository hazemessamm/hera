from functools import partial
from typing import Optional, Tuple, Union

import jax
from jax import lax
from jax import numpy as jnp
from jax.numpy import ndarray


@jax.jit
def linear(inputs: ndarray, weights: ndarray, bias: Optional[ndarray] = None):
    """Applies linear transformation on inputs.

    Args:
        inputs (ndarray): Tensor with shape (*, N).
        weights (ndarray): Tensor with shape (N, M).
        bias (ndarray, optional): Tensor with shape (M,).
                                  Defaults to None.

    Returns:
        ndarray: Tensor with shape (*, M).
    """
    out = jnp.matmul(inputs, weights)
    if bias is not None:
        out += bias
    return out


@jax.jit
def embedding(inputs: ndarray, weights: ndarray):
    """Retrives the embeddings for each index in the input tensor.

    Args:
        inputs (ndarray): Tensor with shape (*, N) and dtype `int`.
        weights (ndarray): Embeddings tensor.

    Raises:
        ValueError: If the inputs are not with dtype `int`

    Returns:
        ndarray: Tensor of embeddings with shape (*, M).
    """
    if inputs.dtype not in [jnp.int64, jnp.int32]:
        raise ValueError("`inputs` should be of type int.")
    return weights[inputs]


@partial(jax.jit, static_argnames=["rate", "training"])
def dropout(
    inputs: ndarray,
    weights: ndarray,
    rate: float,
    rng: ndarray,
    training: bool = True,
):
    """Randomly setting some of the input elements to zeros.

    Args:
        inputs (ndarray): An input tensor with shape (*).
        weights (ndarray): Empty tuple.
        rate (float): The dropout probability between 0. and 1.
        rng (ndarray): Seed that will be used for dropout.
        training (bool, optional): If set to `False` no dropout
                                   will be applied. Defaults to True.

    Returns:
        ndarray: Tensor with randomly dropped out elements.
    """
    # Check if the rate is higher than 0.0 to avoid NaNs when dividing.
    if 1.0 > rate > 0.0:
        keep_prob = 1.0 - rate
        # Generate a tensor of booleans where
        # `False` means drop this neuron and `True` means otherwise
        # we keep (1 - rate) of the neurons.
        # (e.g. if the dropout rate is 0.1 then we keep 0.9 of the neurons)
        keep = jax.random.bernoulli(rng, keep_prob, inputs.shape)

        # Check if training is `False` because in that case we will
        # return the same inputs without dropout
        # Also we divide by the rate to scale the weights.
        out = lax.select(
            training, jnp.where(keep, inputs / keep_prob, 0), inputs
        )
        return out
    elif rate == 1.0:
        return jnp.zeros(inputs.shape)
    else:
        return inputs


@jax.jit
def flatten(inputs: ndarray, weights: ndarray):
    """Flattens the inputs while preserving the batch size axis.

    Args:
        inputs (ndarray): An input tensor with shape (*).
        weights (ndarray): An empty tuple.

    Returns:
        ndarray: Flattened tensor.
    """
    batch_dim = inputs.shape[0]
    return jnp.reshape(inputs, (batch_dim, -1))


@jax.jit
def permute(inputs: ndarray, weights: ndarray, permute_to: Tuple):
    """Permutes the inputs.

    Args:
        inputs (ndarray): An input tensor with shape (*).
        weights (ndarray): An empty tuple.
        permute_to (Tuple): Target axis order.

    Returns:
        ndarray: Permuted tensor.
    """
    return jnp.transpose(inputs, permute_to)


@jax.jit
def global_max_pooling_1d(inputs: ndarray, weights: ndarray):
    """Applies global max pooling on the temporal data.

    Args:
        inputs (ndarray): A 3D input tensor.
        weights (ndarray): An empty tuple.

    Returns:
        ndarray: 2D tensor.
    """
    return jnp.max(inputs, axis=1)


@jax.jit
def global_avg_pooling_1d(inputs, weights):
    """Applies global average pooling on the temporal data.

    Args:
        inputs (ndarray): A 3D input tensor.
        weights (ndarray): An empty tuple.

    Returns:
        ndarray: 2D tensor.
    """
    return jnp.mean(inputs, axis=1)


@jax.jit
def global_max_pooling_2d(inputs, weights):
    """Applies global max pooling on the spatial data.

    Args:
        inputs (ndarray): A 4D input tensor.
        weights (ndarray): An empty tuple.

    Returns:
        ndarray: 2D tensor.
    """
    return jnp.max(inputs, axis=(1, 2))


@jax.jit
def global_avg_pooling_2d(inputs, weights):
    """Applies global max pooling on the spatial data.

    Args:
        inputs (ndarray): A 3D input tensor.
        weights (ndarray): An empty tuple.

    Returns:
        ndarray: 2D tensor.
    """
    return jnp.mean(inputs, axis=(1, 2))


@jax.jit
def layer_normalization(
    inputs: ndarray,
    gamma: ndarray = None,
    beta: ndarray = None,
    eps: float = 1e-05,
):
    """Applies layer normalization on an input tensor

    Args:
        inputs (ndarray): Tensor with shape (*, N)
        gamma (ndarray, optional): Tensor with input (N,). Defaults to None.
        beta (ndarray, optional): Tensor with input (N,). Defaults to None.
        eps (ndarray, optional): Epsilon value that will be added
                                 to avoid division by zero. Defaults to 1e-05.

    Returns:
        ndarray: Tensor with shape (*, N)
    """
    mean = jnp.mean(inputs, axis=-1, keepdims=True)
    var = jnp.mean((inputs - mean) ** 2, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    out = (inputs - mean) / std
    if gamma is not None:
        out *= gamma
    if beta is not None:
        out += beta
    return out


@jax.jit
def create_causal_mask(x: ndarray):
    """Creates causal mask tensor.

    Args:
        x (ndarray): Tensor with shape (*)

    Returns:
        ndarray: Boolean tensor with shape (*)
    """
    causal_mask = jnp.tril(x)
    return (causal_mask != 0).astype("float32")


@jax.jit
def positional_embedding(inputs: ndarray, weights: ndarray):
    """Creates a tensor of positions then retrives the positions embeddings.

    Args:
        inputs (ndarray): An input tensor with
                          shape (batch_size, sequence_length)
        weights (ndarray): Embeddings tensor

    Returns:
        ndarray: Tensor of embeddings for each position.
    """
    batch_size, seq_len = inputs.shape
    positions = jnp.arange(0, seq_len)
    positions = jnp.expand_dims(positions, axis=0)
    positions = jnp.repeat(positions, batch_size, axis=0)
    out = embedding(weights, positions)
    return out


def conv1d(
    inputs: ndarray,
    weights: ndarray,
    bias: Optional[ndarray] = None,
    strides: Tuple = (1,),
    padding: Union[Tuple, str] = "valid",
):
    """Applies 1D Convolution on inputs.

    Args:
        inputs (ndarray): 3D tensor with shape
                          (batch_size, timesteps, embed_dim)
        weights (ndarray): 3D tensor with shape
                           (kernel_size, in_channels, out_channels)
        bias (Optional[ndarray], optional): 1D tensor with shape
                                            (out_channels,). Defaults to None.
        strides (Tuple, optional): Tuple with length 1 indicates the number of
                                   strides. Defaults to (1,).
        padding (Union[Tuple, str], optional): Available options:
                                               ("valid", "same", tuple).
                                               Defaults to "valid".

    Returns:
        ndarray: 3D Tensor with shape (batch_size, timesteps, out_channels)
    """
    dims_spec = ("NHC", "HIO", "NHC")
    dimension_numbers = lax.conv_dimension_numbers(
        (1, 1, int(inputs.shape[-1])), weights.shape, dims_spec
    )
    out = lax.conv_general_dilated(
        lhs=inputs,
        rhs=weights,
        window_strides=strides,
        padding=padding,
        dimension_numbers=dimension_numbers,
    )
    if bias is not None:
        out += bias

    return out


def conv2d(inputs, weights, bias=None, strides=(1, 1), padding="valid"):
    """Applies 2D Convolution on inputs.

    Args:
        inputs (ndarray): 4D tensor with shape
                          (batch_size, height, width, in_channels)
        weights (ndarray): 3D tensor with shape
                           (kernel_size, height, width, out_channels)
        bias (Optional[ndarray], optional): 2D tensor with shape
                                            (out_channels,). Defaults to None.
        strides (Tuple, optional): Tuple with length 2 indicates the number of
                                   strides. Defaults to (1, 1).
        padding (Union[Tuple, str], optional): Available options:
                                               ("valid", "same", tuple).
                                               Defaults to "valid".

    Returns:
        ndarray: 4D Tensor with shape (batch_size, height, width, out_channels)
    """
    dims_spec = ("NHWC", "HWIO", "NHWC")
    dimension_numbers = lax.conv_dimension_numbers(
        inputs.shape, weights.shape, dims_spec
    )

    out = lax.conv_general_dilated(
        lhs=inputs,
        rhs=weights,
        window_strides=strides,
        padding=padding,
        dimension_numbers=dimension_numbers,
    )
    if bias is not None:
        out += bias

    return out


def batch_normalization(inputs, gamma, beta):
    pass


def lstm(inputs, weights, bias):
    pass


def gru(inputs, weights, bias):
    pass
