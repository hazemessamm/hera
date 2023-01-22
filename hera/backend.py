import jax
from jax import tree_util
from jax.lib import xla_bridge

class PyTreeRegisterationStateError(Exception):
    pass


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
            setattr(cls.instance, '_state', False)
            setattr(cls.instance, 'registered_modules', set())
        return cls.instance

def enable_auto_register(state: bool):
    if not isinstance(state, bool):
        raise ValueError(
            "Expected `state` to be boolean. " f"Recieved {type(state)}."
        )
    if _PyTreeRegisterationState.instance is not None:
        raise PyTreeRegisterationStateError("Cannot change the state of the "
                                            "pytree class registeration "
                                            "after constructing a module. "
                                            "Please change the state before "
                                            "you create any module.")
    instance = _PyTreeRegisterationState()
    instance._state = state

def auto_register_enabled():
    return _PyTreeRegisterationState()._state

def register_module_if_pytrees_enabled(cls):
    instance = _PyTreeRegisterationState()
    if (
        auto_register_enabled()
        and cls.__name__ not in instance.registered_modules
    ):
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
