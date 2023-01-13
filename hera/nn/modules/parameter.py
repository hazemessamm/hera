from jax import tree_util

@tree_util.register_pytree_node_class
class Parameter:
    _reconstructed_from_unflatten = False
    def __init__(self, rng=None, initializer=None, shape=None):
        self._data = None
        self._name = None
        
        # skip init
        if self._reconstructed_from_unflatten:
            return
        
        self.rng = rng
        self.initializer = initializer
        self.shape = shape

    @property
    def data(self):
        if self._data is None:
            self.reset_parameter()
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value

    def reset_parameter(self):
        if self.shape is None:
            self._data = ()
        else:
            self._data = self.initializer(self.rng, self.shape)

    def update_parameter(self, new_data, trainable=True):
        if isinstance(new_data, tuple) and len(new_data) == 0:
            return
        if trainable:
            if new_data.shape == self._data.shape:
                self._data = new_data
            else:
                raise ValueError(
                    f"Shape mismatch. {new_data.shape} != {self._data.shape}"
                )

    def tree_flatten(self):
        aux = (self.rng, self.initializer, self.shape, self._name)
        children = (self._data,)
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(*aux_data[:-1])
        cls._reconstructed_from_unflatten = True
        obj._name = aux_data[-1]
        obj._data = children[0]
        cls._reconstructed_from_unflatten = False
        return obj
