from hera.nn.modules.module import Module

class ModuleList(Module):
    def __init__(self, modules, jit=False):
        super().__init__(jit=jit)
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)

    def __iter__(self):
        for mod in self.nested_modules:
            yield mod
