import jax
from jax.lib import xla_bridge
import contextlib
import h5py
from collections import OrderedDict
from functools import wraps
import warnings
import threading


class _ModuleState:
    """Tracker class for all modules.

    Raises:
        ValueError: If the passed `state` is not a boolean.

    Returns:
        _ModuleState: The first initialized instance (Singleton Pattern).
    """
    instance = None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super(_ModuleState, cls).__new__(cls)
            setattr(cls.instance, 'tracking', False)
        return cls.instance

class _RNGState:
    instance = None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super(_RNGState, cls).__new__(cls)
            setattr(cls.instance, 'rngs', {})
        return cls.instance
    
    def add_global_rng(self, name, rng):
        if name not in self.rngs:
            self.rngs[name] = rng
        else:
            raise ValueError(f"Global rng with name {name} already exists.")
    
    def update_global_rng(self, name):
        self.rngs[name], _ = jax.random.split(self.rngs[name])


def global_rng():
    instance = _RNGState()
    return instance.rngs.get('global_rng', None)

def set_global_rng(rng):
    instance = _RNGState()
    if isinstance(rng, int):
        rng = jax.random.PRNGKey(rng)
        
    instance.add_global_rng('global_rng', rng)


def get_and_update_global_rng():
    isinstance = _RNGState()
    rng = isinstance.rngs['global_rng']
    isinstance.update_global_rng('global_rng')
    return rng


def platform():
    return xla_bridge.get_backend().platform


def is_gpu_available():
    return xla_bridge.get_backend().platform == "gpu"


def devices():
    return jax.devices()


def device_count():
    return jax.device_count()


def mark_deprecated(use_instead: str = None):
    def _mark_deprecated(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"{func.__name__} is deprecated and will be removed soon."
            if use_instead is not None:
                msg += f"Use `{use_instead}` instead."
            warnings.warn(msg, DeprecationWarning)
            out = func(*args, **kwargs)
            return out

        return wrapper
    return _mark_deprecated


def mark_experimental(use_instead: str = None):
    def _mark_experimental(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"`{func.__name__}` is still experimental."
            if use_instead is not None:
                msg += f" Use `{use_instead}` instead for stability."
            warnings.warn(msg, UserWarning)
            out = func(*args, **kwargs)
            return out
        return wrapper
    return _mark_experimental


@contextlib.contextmanager
def track_tensor():
    """Context manager that enables tracking of tensors inside modules.

    Example:
        >>> with hera.backend.track_tensor():
       ...     self.weight = initializers.zeros(key, shape)
    """
    instance = _ModuleState()
    try:
        instance.tracking = True
        yield None
    finally:
        instance.tracking = False


def tracking():
    instance = _ModuleState()
    return instance.tracking


def create_keys(rng, num_keys):
    return jax.random.split(rng, num_keys)


def save_weights(module, prefix, extention='.h5'):
        if not prefix.endswith(extention):
            prefix += extention

        with h5py.File(prefix, 'w') as f:
            f.update(module.state_dict())


def load_weights(prefix, extention='.h5'):
    weights = OrderedDict()
    if not prefix.endswith(extention):
        prefix += extention

    with h5py.File(prefix, 'r') as f:
        for k, v in f.items():
            weights[k] = v[:]
    return weights
