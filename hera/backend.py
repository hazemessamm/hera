import jax
from jax import tree_util
from jax.lib import xla_bridge


class _PyTreeRegisterationState:
    """Tracker class for the registered modules if PyTree registeration enabled.

    Raises:
        ValueError: If the passed `state` is not a boolean.

    Returns:
        _PyTreeRegisterationState: The first initialized instance (Singleton Pattern).
    """
    instance = None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super(_PyTreeRegisterationState, cls).__new__(cls)
        return cls.instance

    def __init__(self, state: bool):
        if not isinstance(state, bool):
            raise ValueError("Expected `state` to be boolean. "
                             f"Recieved {type(state)}.")
        self.state = state
        self.registered_modules = set()


def enable_auto_register(state: bool):
    if not isinstance(state, bool):
        raise ValueError("Expected `state` to be boolean. "
                         f"Recieved {type(state)}.")
    _PyTreeRegisterationState(state)


def auto_register_enabled():
    instance = _PyTreeRegisterationState.instance
    if instance is None:
        return False
    else:
        return instance.state

def register_module_if_pytrees_enabled(cls):
    instance = _PyTreeRegisterationState.instance
    if auto_register_enabled() and cls.__name__ not in instance.registered_modules:
        instance.registered_modules.add(cls.__name__)
        tree_util.register_pytree_node_class(cls)

def platform():
    return xla_bridge.get_backend().platform

def is_gpu_available():
    return xla_bridge.get_backend().platform == "gpu"

def devices():
    return jax.devices()

def device_count():
    return jax.device_count()