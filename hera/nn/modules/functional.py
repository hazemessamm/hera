from functools import partial
from typing import Optional, Tuple, Union

import jax
from jax import lax
from jax import numpy as jnp
from jax.numpy import ndarray
from hera import backend


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
    rate: float,
    rng: ndarray,
    training: bool = True,
):
    """Randomly setting some of the input elements to zeros.

    Args:
        inputs (ndarray): An input tensor with shape (*).
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
def flatten(inputs: ndarray):
    """Flattens the inputs while preserving the batch size axis.

    Args:
        inputs (ndarray): An input tensor with shape (*).

    Returns:
        ndarray: Flattened tensor.
    """
    batch_dim = inputs.shape[0]
    return jnp.reshape(inputs, (batch_dim, -1))


@jax.jit
def permute(inputs: ndarray, permute_to: Tuple):
    """Permutes the inputs.

    Args:
        inputs (ndarray): An input tensor with shape (*).
        permute_to (Tuple): Target axis order.

    Returns:
        ndarray: Permuted tensor.
    """
    return jnp.transpose(inputs, permute_to)


@jax.jit
def global_max_pooling_1d(inputs: ndarray):
    """Applies global max pooling on the temporal data.

    Args:
        inputs (ndarray): A 3D input tensor.

    Returns:
        ndarray: 2D tensor.
    """
    return jnp.max(inputs, axis=1)


@jax.jit
def global_avg_pooling_1d(inputs: ndarray):
    """Applies global average pooling on the temporal data.

    Args:
        inputs (ndarray): A 4D input tensor.

    Returns:
        ndarray: 2D tensor.
    """
    return jnp.mean(inputs, axis=1)


@jax.jit
def global_max_pooling_2d(inputs: ndarray):
    """Applies global max pooling on the temporal data.

    Args:
        inputs (ndarray): A 4D input tensor.

    Returns:
        ndarray: 2D tensor.
    """
    return jnp.max(inputs, axis=(1, 2))


@jax.jit
def global_avg_pooling_2d(inputs: ndarray):
    """Applies global average pooling on the temporal data.

    Args:
        inputs (ndarray): A 3D input tensor.

    Returns:
        ndarray: 2D tensor.
    """
    return jnp.mean(inputs, axis=(1, 2))


@jax.jit
def global_max_pooling_2d(inputs: ndarray):
    """Applies global max pooling on the spatial data.

    Args:
        inputs (ndarray): A 4D input tensor.

    Returns:
        ndarray: 2D tensor.
    """
    return jnp.max(inputs, axis=(1, 2))


@jax.jit
def global_avg_pooling_2d(inputs: ndarray):
    """Applies global max pooling on the spatial data.

    Args:
        inputs (ndarray): A 3D input tensor.

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
        out = jnp.add(out, bias)

    return out


@partial(jax.jit, static_argnames=["num_heads", "embed_dim_per_head"])
def transpose_qkv(query, key, value, num_heads, embed_dim_per_head):
    # Reshape `query`, `key` and `values`
    # from shape [batch_size, seq_len, embed_dim]
    # to [batch_size, seq_len, num_heads, embed_dim // num_heads]
    # then change the `num_heads` axis from
    # [batch_size, seq_len, num_heads, embed_dim // num_heads]
    # to [batch_size, num_heads, seq_len, embed_dim // num_heads]
    # using the `transpose` function
    # and the reason for that is that
    # we want to process each head independetly
    # we want to have each head to have [seq_len, embed_dim // num_heads]
    batch_size = query.shape[0]

    query = jnp.reshape(
        query, (batch_size, -1, num_heads, embed_dim_per_head)
    ).transpose((0, 2, 1, 3))
    key = jnp.reshape(
        key, (batch_size, -1, num_heads, embed_dim_per_head)
    ).transpose((0, 2, 1, 3))
    value = jnp.reshape(
        value, (batch_size, -1, num_heads, embed_dim_per_head)
    ).transpose((0, 2, 1, 3))

    return query, key, value


@jax.jit
def masked_fill(mask, fill, inputs):
    out = jnp.select(mask, inputs, jax.lax.broadcast(fill, inputs.shape))
    return out


@partial(jax.jit, static_argnames=["use_causal_mask"])
def attention(query, key, use_causal_mask):
    # Change the key axis from
    # [batch_size, num_heads, seq_len, embed_dim // num_heads]
    # to [batch_size, num_heads, embed_dim // num_heads, seq_len]
    # and the reason for that is because we need to be able
    # to perform matrix multiplication and in matrix multiplication
    # we need the first matrix columns to match the second matrix rows
    # in our situation we want query shape to be
    # [batch_size, num_heads, seq_len, embed_dim // num_heads]
    # and key shape to be
    # [batch_size, num_heads, embed_dim // num_heads, seq_len]
    # so we can have the last two axis matching each other.
    embed_dim_per_head = query.shape[-1]
    mul_key = key.transpose(0, 1, 3, 2)

    # Scale the scores output by the square root
    # of the embed_dim // num_heads (`embed_dim_per_head`)
    # to avoid having large variance
    scores = jnp.matmul(query, mul_key) / jnp.sqrt(embed_dim_per_head)

    # Apply causal mask if it's set to `True`
    scores = lax.select(
        use_causal_mask,
        masked_fill(create_causal_mask(scores), -jnp.inf, scores),
        scores,
    )
    scores = jax.nn.softmax(scores, axis=-1)
    return scores


@partial(jax.jit, static_argnames=["embed_dim"])
def score_value_matmul_and_transpose_scores(scores, value, embed_dim):
    # Changing back the `out` shape from
    # [batch_size, num_heads, seq_len, embed_dim // num_heads]
    # to [batch_size, seq_len, num_heads, embed_dim // num_heads]
    # so we can collapse back the `num_heads` and `embed_dim // num_heads`
    # together back to `embed_dim`
    batch_size = scores.shape[0]
    scores = jnp.matmul(scores, value)
    scores = jnp.transpose(scores, (0, 2, 1, 3))
    scores = jnp.reshape(scores, (batch_size, -1, embed_dim))
    return scores


@partial(jax.jit, static_argnames=['pool_size', 'strides', 'padding'])
def average_pooling_2d(inputs, pool_size, strides, padding):
    pool_size = (1, *pool_size, 1)
    strides = (1, *strides, 1)

    out = lax.reduce_window(inputs, 0.0, lax.add, pool_size, strides, padding)
    ones = jnp.ones(inputs.shape, dtype=inputs.dtype)
    window_sizes = lax.reduce_window(ones, 0.0, lax.add, pool_size, strides, padding)
    return lax.div(out, window_sizes)


@backend.mark_experimental(use_instead='hera.nn.MultiHeadAttention')
@jax.jit
def multihead_attention(query: jax.numpy.ndarray,
                        key: jax.numpy.ndarray,
                        value: jax.numpy.ndarray,
                        query_weights: jax.numpy.ndarray,
                        key_weights: jax.numpy.ndarray,
                        value_weights: jax.numpy.ndarray,
                        output_weights: jax.numpy.ndarray,
                        num_heads: int,
                        dropout_prob: float,
                        dropout_rng: jax.numpy.ndarray,
                        use_causal_mask: bool,
                        training: bool) -> jax.numpy.ndarray:
    
    embed_dim_per_head = query.shape[-1] // num_heads
    query = linear(query, query_weights)
    key = linear(key, key_weights)
    value = linear(value, value_weights)
    query, key, value = transpose_qkv(
        query, key, value, num_heads, embed_dim_per_head
    )
    scores = attention(query, key, use_causal_mask)
    scores = dropout(scores, dropout_prob, dropout_rng, training)
    scores = score_value_matmul_and_transpose_scores(scores, value, self.embed_dim)
    out = linear(scores, output_weights)
    return out



def batch_normalization(inputs, gamma, beta):
    pass


def lstm(inputs, weights, bias):
    pass


def gru(inputs, weights, bias):
    pass
