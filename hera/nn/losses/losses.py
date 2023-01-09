from hera.nn.modules.module import Module
import optax
from jax.numpy import ndarray
from hera.nn.losses.base_loss import Loss
import jax.numpy as jnp


# TODO
class BCELoss(Loss):
    def forward(self, y_pred, y_true):
        pass


class BCELossWithLogits(Loss):
    def __init__(self, reduction: str = "mean", jit: bool = False):
        super().__init__(reduction, jit)

    def forward(self, y_pred, y_true):
        loss = optax.sigmoid_binary_cross_entropy(y_pred, y_true)
        return self.reduction_function(loss)


class SparseCrossEntropyLoss(Loss):
    def __init__(self, reduction: str = "mean", jit: bool = False):
        super().__init__(reduction, jit)

    def _apply_sparse_cross_entropy_loss(
        self, y_pred: ndarray, y_true: ndarray
    ):
        loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, y_true)
        return loss

    def forward(self, y_pred: ndarray, y_true: ndarray):
        loss = self._apply_sparse_cross_entropy_loss(y_pred, y_true)
        return self.reduce_loss(loss)


class CrossEntropyLoss(Loss):
    def __init__(self, reduction: str = "mean", jit: bool = False):
        super().__init__(reduction, jit)

    def _apply_cross_entropy_loss(self, y_pred: ndarray, y_true: ndarray):
        loss = optax.softmax_cross_entropy(y_pred, y_true)
        return loss

    def forward(self, y_pred: ndarray, y_true: ndarray):
        loss = self._apply_cross_entropy_loss(y_pred, y_true)
        return self.reduce_loss(loss)


# TODO
class NLLLoss(Module):
    def forward(self, y_pred, y_true):
        pass


class MSELoss(Loss):
    def __init__(self, reduction: str = "mean", jit: bool = False):
        super().__init__(reduction, jit)

    def _apply_mse(self, y_pred, y_true):
        return (y_true - y_pred) ** 2

    def forward(self, y_pred, y_true):
        loss = self._apply_mse(y_pred, y_true)
        return self.reduce_loss(loss)


class MAELoss(Loss):
    def __init__(self, reduction: str = None, jit: bool = False):
        super().__init__(reduction, jit)

    def _apply_mae(self, y_pred, y_true):
        return jnp.abs(y_true - y_pred)

    def forward(self, y_pred, y_true):
        loss = self._apply_mae(y_pred, y_true)
        return self.reduce_loss(loss)
