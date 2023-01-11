from typing import List

from hera.nn.modules.list_container_base import ListContainerModule


class Sequential(ListContainerModule):
    def __init__(self, modules: List = None, jit: bool = False):
        super().__init__(modules=modules, jit=jit)

    def forward(self, inputs):
        out = inputs
        for mod in self.nested_modules:
            out = mod(out)
        return out

    def forward_with_external_weights(self, weights, inputs):
        out = inputs
        for weight, mod in zip(weights.values(), self.nested_modules):
            out = mod(weight, out)
        return out
