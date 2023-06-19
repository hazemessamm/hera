from typing import Dict, List

from jax.numpy import ndarray
from hera.nn.modules import Module
from collections import OrderedDict


class Sequential(Module):
    def __init__(self, modules: List = None, jit: bool = False):
        super().__init__(jit=jit)
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)
    
    def __getitem__(self, idx):
        return list(self.nested_modules.values())[idx]
    
    def update_parameters(self, new_weights: OrderedDict):
        for k, v in new_weights.items():
            self.nested_modules[k].update_parameters(v)

    def forward(self, weights: Dict, inputs: ndarray):
        out = inputs
        for weight, mod in zip(weights.values(), self.nested_modules.values()):
            out = mod(weight, out)
        return out
