import abc
from collections import OrderedDict
from typing import Any, List, Tuple, Union

import jax
from hera.nn.modules.parameter import Parameter


class Module(abc.ABC):
    def __init__(
        self,
        rng: Union[int, jax.random.PRNGKey] = None,
        stochastic_module: bool = False,
        jit: bool = False,
    ):
        """Base Module

        Args:
            rng (Union[int, jax.random.PRNGKey], optional): Seed or default
                                                            random number for
                                                            initialization.
                                                            if set to `None`
                                                            then this means
                                                            that this module
                                                            does not need it.
                                                            Defaults to None.
            stochastic_module (bool, optional): If set to `True` then this
                                                modules will change the seed
                                                with every call
                                                (e.g. Dropout needs it to drop
                                                different neurons every call)
                                                otherwise it will use the same
                                                random number.
                                                Defaults to False.
            jit (bool, optional): If set to `True` then the `forward()` method
                                  will be wrapped by `jax.jit()` function.
                                  if set to `False` it will remain
                                  a bound method. Defaults to False.
        """

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
        """Resets (re-intializes) all of the nested modules."""
        for mod in self.nested_modules:
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
            elif hasattr(mod, "reset_parameter"):
                mod.reset_parameter()

    def update_parameters(self, new_weights: OrderedDict):
        """Updates the weights of the nested modules.

        Args:
            new_weights (OrderedDict): A dictionary of the same name of the
                                       attributes holding the nested modules
                                       as keys and with weights as values.
        """
        for mod, w in new_weights.items():
            out = getattr(self, mod)
            if isinstance(out, Parameter):
                out.update_parameter(w, self.trainable)
            else:
                out.update_parameters(w)

    def parameters(self):
        """Returns the nested modules parameters.

        Returns:
            Dict: A dictionary of the same name of the attributes holding the
                  nested modules as keys and with weights as values
        """
        return self.state_dict()

    def load_state_dict(self, state: OrderedDict):
        """Loads the state of the module and its nested modules.

        Args:
            state (OrderedDict): State with the attribute names as keys
                                 and tensors as values.
        """
        for k, v in state.items():
            out = getattr(self, k)
            if isinstance(out, Parameter):
                out.data = v
            else:
                out.load_state_dict(v)

    def state_dict(self):
        """Returns the current state of the module and its nested modules.

        Returns:
            Dict: Attribute names as keys and tensors as values.
        """
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
        """Changes the behaviour of the module for evaluation."""
        if self._training:
            self.training = False

    def train(self):
        """Changes the behaviour of the module for training."""
        if not self._training:
            self.training = True

    def reset_rng(self):
        """Resets the current seed to the initial seed.

        Raises:
            ValueError: If the module is not stochastic.
        """
        if self.stochastic_module:
            self.rng = self._initial_rng
        else:
            raise ValueError(f"{self} is not a stochastic module.")

    def compute_output_shape(self, input_shape: Tuple):
        """Utiliity function to calculate the output of a module without
           calling the forward function.

        Args:
            input_shape (Tuple): A tuple with example input shape
                                 (Add a dummy batch_size in the first index.).

        Returns:
            Tuple: Output shape.
        """
        return input_shape

    def create_keys(self, n):
        """Creates N keys from the initial seed.

        Args:
            n (int): Number of keys to be created.

        Returns:
            ndarray: A tensor with N keys.
        """
        return jax.random.split(self.rng, n)

    def make_random_key(self):
        """Returns a new random key each time it gets called.

        Returns:
            ndarray: A new key.
        """
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
                msg = """super().__init__() is not called.
                         Please add it in your subclass `__init__`
                      """
                raise Exception(msg)
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

    def __call__(self, weights, *args, **kwargs):
        """Calls the pre_forward (if implemented) and the forward function.

        Returns:
            ndarray: An output tensor.
        """
        self.jit_forward()
        out = self.forward(weights, *args, **kwargs)

        return out
