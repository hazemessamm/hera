import abc
from collections import OrderedDict
from typing import Any, List, Union

import jax

from hera.nn.modules.parameter import Parameter


class Module(abc.ABC):
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._internal_args = args
        obj._internal_kwargs = kwargs

        # Not a very wise solution,
        # more clear solution is to use meta-classes
        # TODO: Add meta-class to control the module class.
        try:
            jax.tree_util.register_pytree_node_class(cls)
        except ValueError:
            pass
        return obj

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
        if len(self.nested_modules) > 0:
            for mod in filter(
                lambda x: isinstance(x, Module) and x.non_deterministic,
                self.nested_modules,
            ):
                mod.reset_rng()
        else:
            if self.non_deterministic:
                self.rng = self._initial_rng

    def compute_output_shape(self, input_shape):
        inputs = jax.core.ShapedArray(
            (1, *input_shape[1:]), dtype=jax.numpy.float32
        )
        shape = jax.eval_shape(self.forward, inputs).shape
        # To avoid changing the rng while calculating the output shape.
        self.reset_rng()
        return (None, *shape[1:])

    def create_keys(self, n):
        return jax.random.split(self.rng, n)

    def make_random_key(self):
        if self.non_deterministic:
            rng, subkey = jax.random.split(self.rng, 2)

            # To avoid leaks and changing the rng while tracing.
            if not isinstance(rng, jax.core.Tracer):
                self.rng = rng

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
                for mod in filter(
                    lambda x: isinstance(x, Module), self.nested_modules
                ):
                    if any(isinstance(i, Module) for i in mod.nested_modules):
                        mod.jit_forward()
                    else:
                        mod.forward = jax.jit(mod.forward)
            else:
                mod.forward = jax.jit(mod.forward)
            self._jit_compiled = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.jit_forward()

        try:
            pre_out = self.pre_forward_hook(*args, **kwargs)
        except NotImplementedError:
            pre_out = None

        if isinstance(pre_out, tuple):
            args += pre_out
        elif isinstance(pre_out, dict):
            kwargs.update(pre_out)

        out = self.forward(*args, **kwargs)
        return out

    def tree_flatten(self):
        children = tuple(mod for mod in self.nested_modules)
        aux = (self._internal_args, self._internal_kwargs)
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(*aux_data[0], **aux_data[1])

        for i in range(len(obj.nested_modules)):
            attr = getattr(obj, str(obj.nested_modules[i]._name), None)
            if attr is not None:
                setattr(obj, attr._name, children[i])
            else:
                if len(obj.nested_modules) > 0:
                    for i in range(len(obj.nested_modules)):
                        obj.nested_modules[i] = children[i]
                        obj.nested_modules[i]._name = str(i)
        return obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
