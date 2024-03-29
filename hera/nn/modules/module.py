from __future__ import annotations

import abc
import inspect
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import jax
import numpy as np

from hera import backend


def _is_tracer(instance):
    return isinstance(instance, jax.core.Tracer)



class Module(abc.ABC):
    def __init__(
        self,
        rng: Optional[int] = None,
        non_deterministic: bool = False,
        requires_rng: bool = False,
        jit_compile: bool = False,
    ):

        self._rng = rng
        self._requires_rng = requires_rng

        # (e.g. Dropout requires different key
        # each call to drop different neurons.)
        self.non_deterministic = non_deterministic

        self._specify_rng_value()

        self.nested_modules = OrderedDict()
        self.rngs = OrderedDict()
        self._name = None

        self.trainable = True
        self.jit_compile = jit_compile
        self._jit_compiled = False
        self._training = True

    @property
    def rng(self):
        if self._rng is None:
            self._specify_rng_value()
        return self._rng

    def _specify_rng_value(self):
        if (
            self._requires_rng
            and self._rng is None
            and backend.global_rng() is None
        ):
            raise ValueError(
                "`rng` must be either a random number or you "
                "should set a global rng using "
                "`self.set_global_key()` function. "
                f"Received: {self._rng}"
            )

        if self._requires_rng and self._rng is not None:
            self._rng = jax.random.PRNGKey(self._rng)
        elif self._requires_rng:
            self._rng = backend.get_and_update_global_rng()

        if self.non_deterministic:
            self._initial_rng = self._rng

    @property
    def training(self):
        return self._training


    def register_parameter(self, name: str, tensor: jax.numpy.ndarray):
        """Registers a new tensor to the module tensors.

        Args:
            name (str): Name of the tensor.
            tensor (jax.numpy.ndarray): Tensor.
        """
        with backend.track_tensor():
            self.__setattr__(name, tensor)

    def add_weight(
        self,
        rng: jax.random.PRNGKey,
        initializer: callable,
        shape: Tuple,
        name: str,
    ):
        """Creates and adds a tensor and tracks it in a given module

        Args:
            rng (jax.random.PRNGKey): random number for initializing the weight.
            initializer (callable): Initialization function
            shape (Tuple): Tuple that represents the shape of the weight
            name (str): The name of the tensor to be initialized.
        """
        with backend.track_tensor():
            tensor = initializer(rng, shape)
            setattr(self, name, tensor)

    def update_parameters(self, new_weights: OrderedDict):
        """Updated the parameters of each module.

        Args:
            new_weights (OrderedDict): A dictionary of parameters that
                                       will be used to update the
                                       current weights
        """
        for mod, w in new_weights.items():
            out = getattr(self, mod)
            if isinstance(out, jax.numpy.ndarray):
                setattr(self, mod, w)
                self.nested_modules[mod] = w
            else:
                out.update_parameters(w)

    def parameters(self):
        """Returns a dictionary of parameters.

        Returns:
            Dict: Dictionary of parameters.
        """
        out = OrderedDict()
        for mod_name, mod in self.nested_modules.items():
            if isinstance(mod, Module):
                out[mod_name] = mod.parameters()
            elif isinstance(mod, (jax.numpy.ndarray, np.ndarray)):
                out[mod_name] = mod
            else:
                out[mod_name] = ()
        return out

    def _validate_parameter_keys(self, weights: Dict):
        weight_keys = weights.keys()
        current_module_keys = self.nested_modules.keys()

        unknown_keys = weight_keys - current_module_keys
        if len(unknown_keys) > 0:
            raise ValueError(
                "The passed parameters must have the same keys "
                "as the current parameters. "
                f"Unknown keys: {unknown_keys}"
            )

        missing_keys = current_module_keys - weight_keys
        if len(missing_keys) > 0:
            raise ValueError(
                "The passed parameters must have the same keys "
                "as the current parameters. "
                f"Missing keys: {missing_keys}"
            )

    def load_state_dict(self, weights: OrderedDict):
        """Loads the passed parameters to each module.

        Args:
            weights (OrderedDict): Dictionary of parameters.
        """

        self._validate_parameter_keys(weights)
        for k, v in weights.items():
            subkeys = k.split(".")
            mod = self
            for subk in subkeys:
                if subk.isdigit():
                    current_mod = mod[int(subk)]
                else:
                    current_mod = getattr(mod, subk)

                if isinstance(current_mod, Module):
                    mod = current_mod

            mod.nested_modules[subkeys[-1]] = v
            mod.__setattr__(subkeys[-1], v)

    def state_dict(self):
        """Returns a dictionary of parameters for each module.

        Returns:
            Dict: Dictionary of parameters for each module.
        """
        state = OrderedDict()

        def _state_dict(mod, state, prefix):
            for m_name, m in mod.nested_modules.items():
                if isinstance(m, (jax.numpy.ndarray, np.ndarray)):
                    state[prefix + m_name] = m
                else:
                    _state_dict(m, state, prefix + m_name + ".")
            return state

        for mod_name, mod in self.nested_modules.items():
            if isinstance(mod, (jax.numpy.ndarray, np.ndarray)):
                state[mod_name] = mod
            else:
                prefix = mod_name + "."
                state = _state_dict(mod, state, prefix)
        return state

    def train(self, state: bool = True):
        self._training = state
        for mod in filter(
            lambda x: isinstance(x, Module), self.nested_modules.values()
        ):
            mod.train(state=state)

    def eval(self):
        """Switches the module to evaluation mode."""
        self.train(state=False)

    def reset_rng(self):
        """Resets the random number of the module is not deterministic."""
        if len(self.nested_modules) > 0:
            for mod in filter(
                lambda x: isinstance(x, Module) and x.non_deterministic,
                self.nested_modules.values(),
            ):
                mod.reset_rng()
        else:
            if self.non_deterministic:
                self._rng = self._initial_rng

    def compute_output_shape(self, *input_shapes):
        inputs = tuple(
            jax.core.ShapedArray((1, *input_shape[1:]), dtype=jax.numpy.float32)
            for input_shape in input_shapes
        )

        if len(self.nested_modules) == 0 and len(
            inspect.getfullargspec(self.forward).args[1:]
        ) == len(inputs):
            shape = jax.eval_shape(self.forward, *inputs).shape
        else:
            shape = jax.eval_shape(
                self.forward, self.parameters(), *inputs
            ).shape

        # To avoid changing the rng while calculating the output shape.
        self.reset_rng()
        return (None, *shape[1:])

    def make_random_key(self):
        """Creates a different random key each time it gets called for
           non-deterministic modules.

        Returns:
            jax.random.DeviceArray: New random key.
        """
        if self.non_deterministic:
            rng, subkey = jax.random.split(self.rng, 2)

            # To avoid leaks and changing the rng while tracing.
            if not _is_tracer(rng):
                self._rng = rng
            return subkey
        else:
            return self.rng

    def _jit_compile_forward_fn(self):
        """Uses `jax.jit` to compile a given module, it will be called 
           automatically if `jit_compile=True`
           while the instantiation of the instance.
        """
        if self.jit_compile and isinstance(self, Module):
            nested_mods = [
                mod
                for mod in self.nested_modules.values()
                if isinstance(mod, Module)
            ]
            if not nested_mods:
                self.forward = jax.jit(self.forward)
                self.jit_compile = True
            else:
                for mod in nested_mods:
                    mod._jit_compile_forward_fn()

    def check_if_init_called(self):
        """Check if `__init__` is called before performing any operation.

        Raises:
            Exception: An exception is raised if `__init__` is not called.
        """
        if not hasattr(self, "nested_modules"):
            raise Exception(
                "super().__init__() is not called. "
                "Please add it in your subclass `__init__`"
            )

    def add_module(self, name: str, module: Module):
        """Tracks a module by adding it to `nested_modules` attribute.

        Args:
            name (str): Name of the module.
            module (Module): Module to track.
        """
        self.check_if_init_called()
        self.nested_modules[name] = module
        module._jit_compile_forward_fn()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module):  # Modules are automatically tracked
            self.add_module(name, value)
        elif backend.tracking() and isinstance(value, jax.numpy.ndarray):
            self.check_if_init_called()
            self.nested_modules[name] = value

        super(Module, self).__setattr__(name, value)

    def pre_forward_hook(self, *args, **kwargs):
        return None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Changes the module state to
           evaluation in order to make a prediction.

           >>> model = hera.nn.Linear(10, 10)
           >>> x = jax.random.normal(jax.random.PRNGKey(3), (2, 10))
           >>> model.predict(x) # No need to pass the model parameters.

        Returns:
            jax.numpy.ndarray: Prediction outputs.
        """
        latest_state = self.training
        self.training = False
        args, kwargs = self.call_pre_forward_hook_if_implemented(
            *args, **kwargs
        )
        out = self.forward(self.parameters(), *args, **kwargs)
        self.training = latest_state
        return out

    def call_pre_forward_hook_if_implemented(self, *args, **kwargs):
        """If `pre_forward_hook` is implemented it will get called and
           its outputs will be append to either `args` or `kwargs`

        Returns:
            Tuple: `args` and `kwargs` that were originally passed but
                   appended the `pre_forward_hook` output to it.
        """
        pre_out = self.pre_forward_hook(*args, **kwargs)
        if pre_out is not None:
            if isinstance(pre_out, tuple):
                args += pre_out
            elif isinstance(pre_out, dict):
                kwargs.update(pre_out)
        return args, kwargs

    def __call__(self, *args, **kwargs):
        """Calls the necessary functions before calling `forward()` function

        Returns:
            jax.numpy.ndarray: Prediction outputs.
        """

        args, kwargs = self.call_pre_forward_hook_if_implemented(
            *args, **kwargs
        )

        if len(self.nested_modules) == 0:
            num_args = len(inspect.getfullargspec(self.forward).args[1:])
            if len(args) > num_args or len(args) + len(kwargs) > num_args:
                args = args[1:]

        out = self.forward(*args, **kwargs)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
