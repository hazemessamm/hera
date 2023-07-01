from contextlib import contextmanager
from functools import partial
from typing import Dict, Tuple, Union

import jax
import optax
from jax.numpy import ndarray

from hera.nn import Loss, Module
from hera.nn.optimizers import Optimizer

apply_updates = jax.jit(optax.apply_updates)


@contextmanager
def eval_mode(model):
    current_state = model.training
    try:
        model.training = False
        yield None
    finally:
        model.training = current_state


@partial(jax.value_and_grad, argnums=0, has_aux=True)
def _backward_with_external_weights(
    weights, module, loss, *args, targets, **kwargs
):
    preds = module(weights, *args, **kwargs)
    loss_val = loss(preds, targets)
    return loss_val, preds


class BackwardRecorder:
    def __init__(self,
                 module: Module,
                 loss: Loss,
                 optimizer: Optimizer = None,
                 auto_zero_grad: bool = False):
        """Applies forward and backward propagation.

        Args:
            module (Module): The module that will make the inference and
                             retrieve it s gradients.
            loss (Loss): The loss function that will be used for calculating
                         the gradients.
            optimizer (Optimizer, optional): If the optimizer is passed then
                                             `.apply_updates()` method could be
                                             used to call the optimizer and
                                             update the model weights.
                                             Defaults to None.
            auto_zero_grad (bool): If True then the gradients will be
                                   automatically zeroed out, otherwise it will
                                   be accumulated on the previous gradients.
            
        """
        self.module = module
        self.loss = loss
        self.optimizer = optimizer
        self.gradients = None
        self.auto_zero_grad = auto_zero_grad

    def backward(
        self, *args, targets: Union[Tuple, ndarray], **kwargs
    ) -> Tuple[ndarray, ndarray, Dict]:
        """Applies backward propagation.

        Args:
            targets (Union[Tuple, ndarray]): The labels that will be used in
                                             the loss function.

        Returns:
            Tuple[ndarray, ndarray, Dict]: Returns the loss value, predictions
                                           and the gradients as a dictionary.
        """
        return _backward_with_external_weights(
            self.module.parameters(),
            self.module,
            self.loss,
            *args,
            targets=targets,
            **kwargs,
        )

    def step(self):
        """Update the weights using the assigned optimizer and
           update the model with the new weights.

        Args:
            gradients (Dict): Dictionary of attribute names as keys and
                              gradients as values.
            params (Dict): Dictionary of attribute names as keys and
                           weights as values.

        Raises:
            ValueError: `.apply_updates()` cannot update the weights while not
                        passing the optimizer in `__init__()`
        """
        if self.optimizer is not None:
            self.optimizer.step(gradients=self.gradients)
        else:
            raise ValueError(
                "Expected optimizer with type `Optimizer`. "
                f"Received {self.optimizer}. Pass the optimizer in the __init__"
                " (e.g. __init__(model=model, loss=loss, optimizer=optimizer))"
            )

    def zero_grad(self):
        self.gradients = None

    def _accumulate_gradients(self, gradients):
        for k, v in gradients.items():
            self.gradients[k] += v

    def accumulate_gradients(self, gradients):
        if self.gradients is not None and not self.auto_zero_grad:
            self._accumulate_gradients(gradients=gradients)
        else:
            self.zero_grad()
            self.gradients = gradients

    def __enter__(self, *args, **kwargs):
        return self

    def __call__(self, *args, targets, **kwargs):
        (loss_value, predictions), gradients = self.backward(
            *args, targets=targets, **kwargs
        )
        self.accumulate_gradients(gradients)
        return loss_value, predictions

    def __exit__(self, *args, **kwargs):
        return None
