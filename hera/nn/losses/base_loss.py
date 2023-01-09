from jax import numpy as jnp


class Loss:
    def __init__(self, reduction: str = None, jit: bool = False):
        self.reduction = reduction
        self.set_reduction(reduction=reduction)

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

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
