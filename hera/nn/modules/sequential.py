from typing import Dict, List

import jax
from hera.nn.modules import Module
from collections import OrderedDict


class Sequential(Module):
    def __init__(self, modules: List = None, jit_compile: bool = False):
        super().__init__(jit_compile=jit_compile, requires_rng=False)
        for idx, module in enumerate(modules, start=0):
            self.add_module(str(idx), module)
    
    def __getitem__(self, idx):
        return list(self.nested_modules.values())[idx]
    
    def update_parameters(self, new_weights: OrderedDict):
        for k, v in new_weights.items():
            self.nested_modules[k].update_parameters(v)

    def forward(self, weights: Dict, inputs: jax.numpy.ndarray):
        out = inputs
        for weight, mod in zip(weights.values(), self.nested_modules.values()):
            out = mod(weight, out)
        return out
