import abc
from collections import OrderedDict
from typing import Any, List, Union

import jax

from hera import backend
from hera.nn.modules.parameter import Parameter
import h5py


class Module(abc.ABC):
    _reconstructed_from_unflatten = False
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        if cls._reconstructed_from_unflatten:
            obj._reconstructed_from_tree_unflatten = True
        cls.save_init_params_if_pytrees_enabled(obj, *args, **kwargs)
        backend.register_module_if_pytrees_enabled(cls)
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
        
        # (e.g. Dropout requires different key
        # each call to drop different neurons.)
        self.non_deterministic = non_deterministic
        self._name = None

        # skip init because the 
        # rest of the attributes are not important.
        if self._reconstructed_from_unflatten:
            return

        if self.non_deterministic:
            self._initial_rng = self.rng

        # TODO: if a module already has a function
        # that is JIT compiled then no need
        # to re-JIT compile the forward.
        self.requires_jit_compilation = False
        self.cannot_jit_compile = False

        self.jit = jit
        self._jit_compiled = False
        self.trainable = True
        self._training = True

    @classmethod
    def save_init_params_if_pytrees_enabled(cls, obj, *args, **kwargs):
        if backend.auto_register_enabled():
            obj._internal_args = args
            obj._internal_kwargs = kwargs

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

    def save_weights(self, prefix):
        with h5py.File(prefix + '.h5', 'w') as f:
            f.update(self.parameters())
    
    def load_weights(self, prefix):
        with h5py.File(prefix + '.h5', 'w') as f:
            for mod in self.nested_modules:
                if isinstance(mod, Module):
                    mod.load_state_dict(f[mod._name][:])
                elif isinstance(mod, Parameter):
                    mod.data = f[mod._name][:]
                else:
                    raise KeyError
                

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
                out[mod._name] = mod.state_dict()
            elif isinstance(mod, Parameter):
                out[mod._name] = mod.data
            else:
                out[mod] = ()
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

        if backend.auto_register_enabled():
            shape = jax.eval_shape(self.forward, inputs).shape
        else:
            shape = jax.eval_shape(
                self.forward_manual, self.parameters(), inputs
            ).shape
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

    def _jit_compile(self):
        if self.jit and isinstance(self, Module):
            nested_mods = list(
                filter(lambda x: isinstance(x, Module), self.nested_modules)
            )
            if not nested_mods:
                if backend.auto_register_enabled():
                    self.forward = jax.jit(self.forward)
                else:
                    self.forward_manual = jax.jit(
                        self.forward_manual
                    )
                self.jit = True
            else:
                for mod in nested_mods:
                    mod._jit_compile()

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, (Module, Parameter)):
            if not hasattr(self, "nested_modules"):
                raise Exception(
                    "super().__init__() is not called. "
                    "Please add it in your subclass `__init__`"
                )
            self.nested_modules.append(__value)
            if not self._reconstructed_from_unflatten and isinstance(
                __value, Module
            ):
                __value._jit_compile()
            __value._name = __name

        super().__setattr__(__name, __value)

    def pre_forward_hook(self, weights, *args, **kwargs):
        return None

    # TODO: save each gradient to its corresponding layer.
    def save_gradients(self, gradients):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def forward_manual(self, weights, *args, **kwargs):
        if len(self.nested_modules) == 0:
            return self.forward(*args, **kwargs)
        else:
            raise NotImplementedError

    def predict(self, *args, **kwargs):
        current_state = self.training

        self.training = False
        pre_out = self.pre_forward_hook(*args, **kwargs)

        if isinstance(pre_out, tuple):
            args += pre_out
        elif isinstance(pre_out, dict):
            kwargs.update(pre_out)

        if backend.auto_register_enabled():
            out = self.forward(*args, **kwargs)
        else:
            out = self.forward_manual(
                self.parameters(), *args, **kwargs
            )
        self.training = current_state
        return out

    def __call__(self, *args, **kwargs):
        pre_out = self.pre_forward_hook(*args, **kwargs)

        if isinstance(pre_out, tuple):
            args += pre_out
        elif isinstance(pre_out, dict):
            kwargs.update(pre_out)

        if backend.auto_register_enabled():
            out = self.forward(*args, **kwargs)
        else:
            weights = args[0]
            out = self.forward_manual(
                weights, *args[1:], **kwargs
            )
        return out

    def tree_flatten(self):
        return (self.nested_modules, (self._internal_args, self._internal_kwargs, self._name))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        cls._reconstructed_from_unflatten = True
        obj = cls(*aux_data[0], **aux_data[1])
        obj._name = aux_data[2]
        

        for current_obj, child in zip(obj.nested_modules, children):
            obj.__setattr__(str(current_obj._name), child)
        
        cls._reconstructed_from_unflatten = False
        return obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
