from hera.nn.modules.list_container_base import ListContainerModule


class ModuleList(ListContainerModule):
    def __init__(self, modules, jit=False):
        super().__init__(modules=modules, jit=jit)

    def __iter__(self):
        for mod in self.nested_modules:
            yield mod
