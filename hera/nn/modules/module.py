import abc
from collections import OrderedDict
from typing import Any, List, Union

import jax

from nn.modules.parameter import Parameter


class Module(abc.ABC):
    def __init__(
        self,
        rng: Union[int, jax.random.PRNGKey] = None,
        stochastic_module: bool = False,
        jit: bool = False,
    ):
        if rng is not None:
            if isinstance(rng, int):
                self.rng = jax.random.PRNGKey(rng)
            else:
                self.rng = rng

        self.nested_modules: List[Module] = []
        self.stochastic_module = stochastic_module

        # (e.g. Dropout requires different key
        # each call to drop different neurons.)
        if self.stochastic_module:
            self._initial_rng = self.rng

        self.jit = jit
        self._jitted = False
        self.trainable = True
        self._training = True
        self._name = None
        self._has_weights = True

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, val: bool):
        self._training = val
        for mod in self.nested_modules:
            if isinstance(mod, Module):
                mod.training = val

    def reset_parameters(self):
        for mod in self.nested_modules:
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
            elif hasattr(mod, "reset_parameter"):
                mod.reset_parameter()

    def update_parameters(self, new_weights: OrderedDict):
        for mod, w in new_weights.items():
            out = getattr(self, mod)
            if isinstance(out, Parameter):
                out.update_parameter(w, self.trainable)
            else:
                out.update_parameters(w)

    def parameters(self):
        return self.state_dict()

    def load_state_dict(self, new_weights: OrderedDict):
        for k, v in new_weights.items():
            out = getattr(self, k)
            if isinstance(out, Parameter):
                out.data = v
            else:
                out.load_state_dict(v)

    def state_dict(self):
        out = OrderedDict()
        for mod in self.nested_modules:
            if isinstance(mod, Module):
                out.update({mod._name: mod.state_dict()})
            elif isinstance(mod, Parameter):
                out.update({mod._name: mod.data})
            else:
                out.update({mod: ()})
        return out

    def eval(self):
        if self._training:
            self.training = False

    def train(self):
        if not self._training:
            self.training = True

    def reset_rng(self):
        if self.stochastic_module:
            self.rng = self._initial_rng
        else:
            raise ValueError(f"{self} is not a stochastic module.")

    def compute_output_shape(self, input_shape):
        return input_shape

    def create_keys(self, n):
        return jax.random.split(self.rng, n)

    def make_random_key(self):
        if self.stochastic_module:
            rng, another_key = jax.random.split(self.rng, 2)
            if not isinstance(self.rng, jax.core.Tracer):
                self.rng = rng
            return another_key
        else:
            return self.rng

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, (Module, Parameter)):
            if not hasattr(self, "nested_modules"):
                raise Exception(
                    "super().__init__() is not called. Please add it in your subclass `__init__`"
                )
            self.nested_modules.append(__value)
            __value._name = __name

        super().__setattr__(__name, __value)

    def pre_forward_hook(self, weights, *args, **kwargs):
        raise NotImplementedError

    # TODO: save each gradient to its corresponding layer.
    def save_gradients(self, gradients):
        pass

    def jit_forward(self):
        if self.jit and not self._jitted:
            self.forward = jax.jit(self.forward)
            self._jitted = True

    # @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _forward_with_weights(self, weights, *args, **kwargs):
        try:
            out = self.pre_forward_hook(weights, *args, **kwargs)
            if isinstance(out, dict):
                kwargs.update(out)
            elif isinstance(out, tuple):
                args += out
        except NotImplementedError:
            out = None

        return self.forward(weights, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.jit_forward()

        if self._has_weights:
            weights, *args = args
            out = self._forward_with_weights(weights, *args, **kwargs)
        else:
            out = self.forward(*args, **kwargs)
        
        return out


        
