import jax


@jax.tree_util.register_pytree_node_class
class Parameter:
    def __init__(self, rng, initializer, shape):
        self.data = None
        self.rng = rng
        self.initializer = initializer
        self.shape = shape
        self._name = None

    def reset_parameter(self):
        if self.shape is None:
            self.data = ()
        else:
            self.data = self.initializer(self.rng, self.shape)

    def update_parameter(self, new_data, trainable=True):
        if isinstance(new_data, tuple) and len(new_data) == 0:
            return
        if trainable:
            if new_data.shape == self.data.shape:
                self.data = new_data
            else:
                raise ValueError(
                    f"Shape mismatch. {new_data.shape} != {self.data.shape}"
                )

    def tree_flatten(self):
        aux = (self.rng, self.initializer, self.shape, self._name)
        children = (self.data,)
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(*aux_data[:-1])
        obj._name = aux_data[-1]
        obj.data = children[0]
        return obj
