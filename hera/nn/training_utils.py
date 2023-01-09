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
        yield
    finally:
        model.training = current_state


class BackwardRecorder:
    def __init__(self, module: Module, loss: Loss, optimizer: Optimizer = None):
        """Applies forward and backward propagation.

        Args:
            module (Module): The module that will make the inference and
                             retrieve its gradients.
            loss (Loss): The loss function that will be used for calculating
                         the gradients.
            optimizer (Optimizer, optional): If the optimizer is passed then
                                             `.apply_updates()` method could be
                                             used to call the optimizer and
                                             update the model weights.
                                             Defaults to None.
        """
        self.module = module
        self.loss = loss
        self.optimizer = optimizer

    def backward(
        self, weights: Dict, *args, targets: Union[Tuple, ndarray], **kwargs
    ) -> Tuple[ndarray, ndarray, Dict]:
        """Applies backward propagation.

        Args:
            weights (Dict): Dictionary of attribute names as keys and
                            weights as values.
            targets (Union[Tuple, ndarray]): The labels that will be used in
                                             the loss function.

        Returns:
            Tuple[ndarray, ndarray, Dict]: Returns the loss value, predictions
                                           and the gradients as a dictionary.
        """
        @partial(jax.value_and_grad, argnums=0, has_aux=True)
        def _backward(weights, *args, targets, **kwargs):
            preds = self.module(weights, *args, **kwargs)
            loss_val = self.loss(preds, targets)
            return loss_val, preds

        return _backward(weights, *args, targets=targets, **kwargs)

    def apply_updates(self, gradients: Dict, params: Dict):
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
            new_weights = self.optimizer.update_weights(
                gradients=gradients, params=params
            )
            self.module.update_parameters(new_weights=new_weights)
        else:
            raise ValueError(
                "Expected optimizer with type `Optimizer`. "
                f"Recieved {self.optimizer}. Pass the optimizer in the __init__"
                " (e.g. __init__(model=model, loss=loss, optimizer=optimizer))"
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
