import abc
from collections import OrderedDict
from typing import Any, List, Union

import jax

from hera.nn.modules.parameter import Parameter


class Module(abc.ABC):
    def __init__(
        self,
        rng: Union[int, jax.random.PRNGKey] = None,
        non_deterministic: bool = False,
        jit: bool = False,
    ):
        if rng is not None:
            if isinstance(rng, int):
                self.rng = jax.random.PRNGKey(rng)
            else:
                self.rng = rng

        self.nested_modules: List[Module] = []
        self.non_deterministic = non_deterministic
        # TODO: if a module already has a function
        # that is JIT compiled then no need to re-JIT compile the forward.
        self.requires_jit_compilation = False

        # (e.g. Dropout requires different key
        # each call to drop different neurons.)
        if self.non_deterministic:
            self._initial_rng = self.rng

        self.jit = jit
        self._jit_compiled = False
        self.trainable = True
        self._training = True
        self._name = None

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, val: bool):
        self._training = val
        for mod in filter(lambda x: isinstance(x, Module), self.nested_modules):
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
        if self.non_deterministic:
            self.rng = self._initial_rng
        else:
            raise ValueError(f"{self} is not a non-deterministic module.")

    def compute_output_shape(self, input_shape):
        return input_shape

    def create_keys(self, n):
        return jax.random.split(self.rng, n)

    def make_random_key(self):
        with jax.core.eval_context():
            if self.non_deterministic:
                self.rng, subkey = jax.random.split(self.rng, 2)
                return subkey
            else:
                return self.rng

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, (Module, Parameter)):
            if not hasattr(self, "nested_modules"):
                raise Exception(
                    "super().__init__() is not called. "
                    "Please add it in your subclass `__init__`"
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
        if self.jit and not self._jit_compiled:
            if len(self.nested_modules) > 0:
                for mod in filter(lambda x: isinstance(x, Module),
                                  self.nested_modules):
                    if any(isinstance(i, Module) for i in mod.nested_modules):
                        mod.jit_forward()
                    else:
                        mod.forward = jax.jit(mod.forward)
            else:
                mod.forward = jax.jit(mod.forward)
            self._jit_compiled = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, weights, *args, **kwargs):
        self.jit_forward()

        try:
            pre_out = self.pre_forward_hook(weights, *args, **kwargs)
        except NotImplementedError:
            pre_out = None

        if isinstance(pre_out, tuple):
            args += pre_out
        elif isinstance(pre_out, dict):
            kwargs.update(pre_out)

        out = self.forward(weights, *args, **kwargs)
        return out
