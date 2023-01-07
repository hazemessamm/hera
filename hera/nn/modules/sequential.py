from collections import OrderedDict

from hera.nn.modules.module import Module


class Sequential(Module):
    def __init__(self, modules, jit=False):
        super().__init__(jit=jit)
        if not isinstance(modules, list):
            raise ValueError(
                "`modules` should be a `list` of moudules. "
                f"Recieved type: {type(modules)}"
            )
        self.nested_modules = modules

    def state_dict(self):
        out = OrderedDict()
        for i, mod in enumerate(self.nested_modules):
            out.update({str(i): mod.state_dict()})
        return out

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
