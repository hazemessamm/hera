class _PyTreeRegisterationState:
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
