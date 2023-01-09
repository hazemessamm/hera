from contextlib import contextmanager
import optax
import jax


@contextmanager
def eval_mode(model):
    try:
        model.training = False
        yield
    finally:
        model.training = True


@jax.jit
def apply_updates(params, updates):
    return optax.apply_updates(params=params, updates=updates)
