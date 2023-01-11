from collections import OrderedDict
from typing import List

from hera.nn.modules.module import Module


class ListContainerModule(Module):
    def __init__(self, modules: List = None, jit: bool = False):
        """Base class for classes like ModuleList, Sequential, ModuleDict

        Args:
            modules (List, optional): _description_. Defaults to None.
            jit (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
        """
        super().__init__(jit=jit)
        if modules is not None:
            if not isinstance(modules, list):
                raise ValueError(
                    "`modules` should be a `list` of modules. "
                    f"Recieved type: {type(modules)}"
                )
            elif any(not isinstance(mod, Module) for mod in modules):
                types = [type(mod) for mod in modules]
                raise ValueError(f'Expected a list of modules of type `Module`. Recieved {type(types)}')
            self.nested_modules = modules
        else:
            self.nested_modules = []
        self._initialize_modules()

    def _initialize_modules(self):
        for idx, mod in enumerate(self.nested_modules):
            mod._name = str(idx)

    def add_modules(self, modules):
        if not isinstance(modules, list):
            raise ValueError(
                "Expected `modules` to be a list. "
                f"Recieved: {type(modules)}."
            )
        if any(not isinstance(mod, Module) for mod in modules):
            types_in_list = [type(mod) for mod in modules]
            raise ValueError(
                "Expected the list to have modules of type "
                f"`Module`. Recieved: {types_in_list}."
            )
        else:
            for idx, mod in enumerate(modules, start=len(self.nested_modules)):
                mod._name = str(idx)
                self.nested_modules.append(mod)

    def add(self, module):
        if not isinstance(module, Module):
            raise ValueError(
                "Expected module with type `Module`. "
                f"Recieved {type(module)}"
            )
        module._name = len(self.nested_modules)
        self.nested_modules.append(module)

    def load_state_dict(self, new_weights: OrderedDict):
        for k, v in new_weights.items():
            self.nested_modules[int(k)].load_state_dict(v)

    def update_parameters(self, new_weights):
        for k, v in new_weights.items():
            self.nested_modules[int(k)].update_parameters(v)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(*aux_data[0], **aux_data[1])
        obj._name = aux_data[2]
        obj._reconstructed_from_tree_unflatten = True

        for i, c in enumerate(children):
            obj.nested_modules[i] = c
            obj.nested_modules[i]._name = str(i)
        return obj
