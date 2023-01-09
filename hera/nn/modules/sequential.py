from collections import OrderedDict

from hera.nn.modules.module import Module
from typing import List


class Sequential(Module):
    def __init__(self, modules: List = None, jit: bool = False):
        super().__init__(jit=jit)
        if modules is not None and not isinstance(modules, list):
            raise ValueError(
                "`modules` should be a `list` of modules. "
                f"Recieved type: {type(modules)}"
            )

        if modules is not None:
            self.nested_modules = modules
        else:
            self.nested_modules = []

        self._initialize_modules()

    def _initialize_modules(self):
        for idx, mod in enumerate(self.nested_modules):
            mod._name = str(idx)

    def add(self, module):
        if not isinstance(module, Module):
            raise ValueError('Expected module with type `Module`. '
                             f'Recieved {type(module)}')
        self.nested_modules.append(module)

    def load_state_dict(self, new_weights: OrderedDict):
        for k, v in new_weights.items():
            self.nested_modules[int(k)].load_state_dict(v)

    def forward(self, weights, inputs):
        out = inputs
        for idx, w in weights.items():
            out = self.nested_modules[int(idx)](w, out)
        return out

    def update_parameters(self, new_weights):
        for k, v in new_weights.items():
            self.nested_modules[int(k)].update_parameters(v)
