from jax import numpy as jnp
import jax
import abc


class Loss(abc.ABC):
    def __init__(self, reduction: str = None, jit: bool = False):
        self.reduction = reduction
        self.set_reduction(reduction=reduction)
        self.jit = jit
        self._jit_compiled = False

    def set_reduction(self, reduction):
        if reduction == "mean":
            self.reduction_function = jnp.mean
        elif reduction == "sum":
            self.reduction_function = jnp.sum
        else:
            self.reduction_function = lambda x: x

    def reduce_loss(self, loss):
        return self.reduction_function(loss)

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def jit_forward(self):
        if self.jit and not self._jit_compiled:
            self.forward = jax.jit(self.forward)
            self._jit_compiled = True

    def __call__(self, *args, **kwargs):
        self.jit_forward()
        return self.forward(*args, **kwargs)
