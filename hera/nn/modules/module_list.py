from collections import OrderedDict

from nn.modules.module import Module


class ModuleList(Module):
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

    def __iter__(self):
        for mod in self.nested_modules:
            yield mod

    def update_weights(self, new_weights):
        for k, v in new_weights.items():
            self.nested_modules[int(k)].update_weights(v)
