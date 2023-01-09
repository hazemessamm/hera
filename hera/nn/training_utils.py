from contextlib import contextmanager
import optax
import jax
from hera.nn import Module, Loss
from functools import partial
from typing import Tuple, Union
from jax.numpy import ndarray
from hera.nn.optimizers import Optimizer


apply_updates = jax.jit(optax.apply_updates)


@contextmanager
def eval_mode(model):
    try:
        model.training = False
        yield
    finally:
        model.training = True


class BackwardRecorder:
    def __init__(self, module: Module, loss: Loss, optimizer: Optimizer = None):
        self.module = module
        self.loss = loss
        self.optimizer = optimizer

    def backward(
        self, weights, *args, targets: Union[Tuple, ndarray], **kwargs
    ):
        @partial(jax.value_and_grad, argnums=0, has_aux=True)
        def _backward(weights, *args, targets, **kwargs):
            preds = self.module(weights, *args, **kwargs)
            loss_val = self.loss(preds, targets)
            return loss_val, preds

        return _backward(weights, *args, targets=targets, **kwargs)

    def apply_updates(self, gradients, params):
        if self.optimizer is not None:
            new_weights = self.optimizer.update_weights(
                gradients=gradients, params=params
            )
            self.module.update_parameters(new_weights=new_weights)
        else:
            raise ValueError(
                "Expected optimizer with type `Optimizer`. "
                f"Recieved {self.optimizer}"
            )

    def __enter__(self, *args, **kwargs):
        return self

    def __call__(self, *args, targets, **kwargs):
        (loss_value, predictions), gradients = self.backward(
            self.module.parameters(), *args, targets=targets, **kwargs
        )
        return loss_value, predictions, gradients

    def __exit__(self, *args, **kwargs):
        return None
